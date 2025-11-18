import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import pandas as pd
import math

# ===============================
# CONFIGURACOES E CACHE
# ===============================
st.set_page_config(layout="wide", page_title="Sudoku Solver Pro (Multi-Size)")

@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)

reader = get_ocr_reader()

# ===============================
# UTILITARIOS VISUAIS E GEOMETRIA
# ===============================
def order_points(pts):
    """Ordena coordenadas: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_sudoku_board(image_np):
    """Localiza o tabuleiro e remove perspectiva."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    found_board = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 2000:
            found_board = approx
            break

    h, w = image_np.shape[:2]
    if found_board is None:
        return image_np, np.eye(3), w, h

    pts = found_board.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_np, M, (maxWidth, maxHeight))
    return warped, M, maxWidth, maxHeight

def detect_grid_size_auto(warped_img):
    """
    Tenta estimar se e 4x4, 6x6, 9x9 ou 16x16 baseado em contornos internos.
    Retorna o N estimado (ex: 9).
    """
    if len(warped_img.shape) == 3:
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = warped_img
        
    # Binarizacao agressiva para pegar grades
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
    
    # Acha contornos internos (potenciais celulas)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = warped_img.shape[:2]
    total_area = h * w
    valid_cells = 0
    
    # Conta contornos que tem tamanho de "celula"
    # Numa 9x9, area ~ 1/81 (~1.2%). Numa 4x4 ~ 1/16 (6%).
    for c in contours:
        area = cv2.contourArea(c)
        if area < total_area * 0.001: continue # Ruido
        if area > total_area * 0.2: continue   # Muito grande
        
        # Formato quadrado?
        x, y, cw, ch = cv2.boundingRect(c)
        ratio = float(cw)/ch
        if 0.6 < ratio < 1.4:
            valid_cells += 1
            
    # Heuristica simples baseada na contagem
    # Como contours detectam "buracos", e comum achar o dobro (borda in/out) ou menos (falhas)
    # Vamos usar thresholds suaves
    if valid_cells < 25:
        return 4 # Provavelmente 4x4 (16 celulas)
    elif valid_cells < 49:
        return 6 # Provavelmente 6x6 (36 celulas)
    elif valid_cells < 110:
        return 9 # Provavelmente 9x9 (81 celulas)
    elif valid_cells >= 110:
        return 16 # Gigante
        
    return 9 # Fallback padrao

# ===============================
# OCR E PROCESSAMENTO DE CELULA
# ===============================
def process_cell_for_ocr(cell_img):
    """Limpa e centraliza o digito."""
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cell_img

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)
    roi = thresh[margin_h:h-margin_h, margin_w:w-margin_w]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < (h * w * 0.02): return None

    x, y, w_c, h_c = cv2.boundingRect(c)
    digit = roi[y:y+h_c, x:x+w_c]

    side = max(w_c, h_c) + 20
    centered = np.zeros((side, side), dtype=np.uint8)
    start_x = (side - w_c) // 2
    start_y = (side - h_c) // 2
    centered[start_y:start_y+h_c, start_x:start_x+w_c] = digit
    
    centered = cv2.bitwise_not(centered)
    return cv2.resize(centered, (50, 50), interpolation=cv2.INTER_AREA)

def split_cells_and_ocr(warped_img, n):
    """Divide em N x N e faz OCR com allowlist dinamica."""
    rows = []
    h, w = warped_img.shape[:2]
    cell_h = h // n
    cell_w = w // n
    
    # Cria lista de numeros permitidos string '123...n'
    allow_chars = "".join([str(i) for i in range(1, n+1)]) # Ex: "1234" ou "123456789"

    for i in range(n):
        row_vals = []
        for j in range(n):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            
            cell = warped_img[y1:y2, x1:x2]
            processed = process_cell_for_ocr(cell)
            
            val = 0
            if processed is not None:
                try:
                    results = reader.readtext(
                        processed, 
                        allowlist=allow_chars,
                        decoder='greedy',
                        text_threshold=0.5,
                        low_text=0.3,
                        detail=1
                    )
                    if results:
                        best = max(results, key=lambda x: x[2])
                        text = best[1].strip()
                        if text.isdigit():
                            val = int(text)
                except:
                    pass
            
            if val > n: val = 0
            row_vals.append(val)
        rows.append(row_vals)
    return rows

# ===============================
# SOLVER GENERICO (N x N)
# ===============================
def get_box_dims(n):
    """Retorna dimensoes do bloco (altura, largura) para N."""
    if n == 9: return (3, 3)
    if n == 4: return (2, 2)
    if n == 6: return (2, 3) # Comum em mini-sudoku 6x6 (2 linhas, 3 colunas no box)
    if n == 16: return (4, 4)
    # Fallback generico para quadrado perfeito
    root = int(math.sqrt(n))
    if root * root == n: return (root, root)
    return (1, n) # Pior caso

def is_valid_initial_board(board, n):
    box_h, box_w = get_box_dims(n)
    
    # Linhas
    for i in range(n):
        row = [x for x in board[i] if x != 0]
        if len(row) != len(set(row)): return False, f"Duplicado na linha {i+1}"
    
    # Colunas
    for j in range(n):
        col = [board[i][j] for i in range(n) if board[i][j] != 0]
        if len(col) != len(set(col)): return False, f"Duplicado na coluna {j+1}"
        
    # Blocos
    # Iterar sobre os blocos
    # Quantos blocos na vertical? n // box_h
    # Quantos blocos na horizontal? n // box_w
    for by in range(n // box_h):
        for bx in range(n // box_w):
            box_vals = []
            start_y = by * box_h
            start_x = bx * box_w
            for i in range(start_y, start_y + box_h):
                for j in range(start_x, start_x + box_w):
                    val = board[i][j]
                    if val != 0: box_vals.append(val)
            if len(box_vals) != len(set(box_vals)):
                return False, f"Duplicado no bloco ({by+1},{bx+1})"
                
    return True, "Ok"

def solve_generic(board, n, box_h, box_w):
    find = find_empty(board, n)
    if not find:
        return True
    row, col = find

    for num in range(1, n + 1):
        if is_valid_move(board, num, (row, col), n, box_h, box_w):
            board[row][col] = num
            if solve_generic(board, n, box_h, box_w):
                return True
            board[row][col] = 0
    return False

def is_valid_move(board, num, pos, n, box_h, box_w):
    r, c = pos
    
    # Linha
    for j in range(n):
        if board[r][j] == num and c != j: return False
    
    # Coluna
    for i in range(n):
        if board[i][c] == num and r != i: return False
        
    # Bloco
    start_row = (r // box_h) * box_h
    start_col = (c // box_w) * box_w
    
    for i in range(start_row, start_row + box_h):
        for j in range(start_col, start_col + box_w):
            if board[i][j] == num and (i, j) != pos:
                return False
    return True

def find_empty(board, n):
    for i in range(n):
        for j in range(n):
            if board[i][j] == 0: return (i, j)
    return None

# ===============================
# DESENHO E RECONSTRUCAO
# ===============================
def draw_solution_generic(warped_img, board_init, board_solved, n):
    img_out = warped_img.copy()
    h, w = img_out.shape[:2]
    cell_h = h // n
    cell_w = w // n
    
    # Ajuste de fonte dinamico
    font_scale = (cell_h / 35.0)
    thickness = max(1, int(cell_h / 15.0))
    font = cv2.FONT_HERSHEY_DUPLEX
    
    for i in range(n):
        for j in range(n):
            if board_init[i][j] == 0 and board_solved[i][j] != 0:
                text = str(board_solved[i][j])
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                x = int((j * cell_w) + (cell_w - tw) / 2)
                y = int((i * cell_h) + (cell_h + th) / 2)
                
                # Desenha borda preta para contraste
                cv2.putText(img_out, text, (x, y), font, font_scale, (0,0,0), thickness+3, cv2.LINE_AA)
                # Cor verde principal
                cv2.putText(img_out, text, (x, y), font, font_scale, (0,255,0), thickness, cv2.LINE_AA)
    return img_out

def overlay_solution(original, warped_sol, M, w_dim, h_dim):
    h_orig, w_orig = original.shape[:2]
    
    warped_back = cv2.warpPerspective(warped_sol, M, (w_orig, h_orig), flags=cv2.WARP_INVERSE_MAP)
    
    ones = np.ones((h_dim, w_dim), dtype=np.uint8) * 255
    mask_back = cv2.warpPerspective(ones, M, (w_orig, h_orig), flags=cv2.WARP_INVERSE_MAP)
    
    mask_inv = cv2.bitwise_not(mask_back)
    img_bg = cv2.bitwise_and(original, original, mask=mask_inv)
    img_fg = cv2.bitwise_and(warped_back, warped_back, mask=mask_back)
    
    return cv2.add(img_bg, img_fg)

# ===============================
# INTERFACE
# ===============================
st.title("Sudoku Solver Pro (Multi-Size)")
st.markdown("Suporta 4x4, 6x6 e 9x9. Detecta automaticamente ou permite escolha manual.")

# Sidebar para controles
with st.sidebar:
    st.header("Configuracoes")
    grid_option = st.selectbox("Tamanho da Grade", ["Auto", "9x9", "6x6", "4x4"])
    use_perspective = st.checkbox("Correcao de Perspectiva", value=True)

uploaded_file = st.file_uploader("Carregue a imagem", type=['jpg', 'png', 'jpeg', 'gif'])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_img)
    
    col1, col2 = st.columns(2)
    col1.image(img_np, caption="Original", use_container_width=True)
    
    # Variaveis de estado
    if 'stage' not in st.session_state: st.session_state['stage'] = 0
    
    if st.button("1. Analisar Imagem"):
        with st.spinner("Processando geometria..."):
            # 1. Perspectiva
            if use_perspective:
                warped, M, wm, hm = find_sudoku_board(img_np)
            else:
                hm, wm = img_np.shape[:2]
                warped, M, wm, hm = img_np, np.eye(3), wm, hm
            
            # 2. Detectar Tamanho (N)
            if grid_option == "Auto":
                detected_n = detect_grid_size_auto(warped)
                st.info(f"Tamanho detectado automaticamente: {detected_n}x{detected_n}")
                n = detected_n
            else:
                n = int(grid_option[0]) # Pega '9' de '9x9'
                
            # 3. OCR
            grid_read = split_cells_and_ocr(warped, n)
            
            # Salvar no estado
            st.session_state.update({
                'original': img_np,
                'warped': warped,
                'M': M,
                'dims': (wm, hm),
                'n': n,
                'grid_read': grid_read,
                'stage': 1
            })
            
            col2.image(warped, caption=f"Processada ({n}x{n})", use_container_width=True)

    # Estagio de Edicao e Solucao
    if st.session_state['stage'] >= 1:
        st.divider()
        st.subheader("2. Verificacao e Solucao")
        
        n = st.session_state['n']
        df = pd.DataFrame(st.session_state['grid_read'])
        
        # Editor
        st.caption("Corrija os numeros se necessario (0 = vazio).")
        edited_df = st.data_editor(df, use_container_width=True, height=(n*35)+50)
        
        if st.button("3. Resolver"):
            final_grid = edited_df.values.tolist()
            
            # Validar
            valid, msg = is_valid_initial_board(final_grid, n)
            if not valid:
                st.error(f"Erro no tabuleiro: {msg}")
            else:
                board_copy = [r[:] for r in final_grid]
                box_h, box_w = get_box_dims(n)
                
                with st.spinner("Resolvendo..."):
                    solved = solve_generic(board_copy, n, box_h, box_w)
                    
                if solved:
                    st.success(f"Resolvido! (Modo {n}x{n})")
                    
                    # Desenhar
                    res_warped = draw_solution_generic(
                        st.session_state['warped'],
                        final_grid,
                        board_copy,
                        n
                    )
                    
                    res_final = overlay_solution(
                        st.session_state['original'],
                        res_warped,
                        st.session_state['M'],
                        st.session_state['dims'][0],
                        st.session_state['dims'][1]
                    )
                    
                    st.image(res_final, caption="Solucao Final", use_container_width=True)
                else:
                    st.error("Sem solucao possivel para a configuracao atual.")
