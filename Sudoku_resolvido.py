import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

# -------------------------------
# Configuração do EasyOCR
# -------------------------------
reader = easyocr.Reader(["en"], gpu=False)

# -------------------------------
# Configuração da Página
# -------------------------------
st.set_page_config(layout="wide", page_title="Sudoku Solver com EasyOCR")
st.title("Resolvedor de Sudoku via EasyOCR")

uploaded = st.file_uploader("📸 Envie uma imagem do Sudoku", type=["png", "jpg", "jpeg"])

# -------------------------------
# Funções auxiliares
# -------------------------------
def find_empty(board):
    """Encontra a próxima célula vazia (0)"""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return (i, j)
    return None

def valid(board, num, pos, box):
    """Verifica se um número é válido em uma posição"""
    row, col = pos
    n = len(board)

    # Linha
    if num in board[row]:
        return False

    # Coluna
    if num in [board[i][col] for i in range(n)]:
        return False

    # Subgrade (3x3)
    box_x = col // box
    box_y = row // box
    for i in range(box_y * box, box_y * box + box):
        for j in range(box_x * box, box_x * box + box):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board):
    """Resolve o Sudoku usando backtracking"""
    n = len(board)
    box = int(np.sqrt(n))
    empty = find_empty(board)
    if not empty:
        return True

    row, col = empty
    for num in range(1, n + 1):
        if valid(board, num, (row, col), box):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

def ocr_sudoku_easyocr(image, n):
    """Lê números do Sudoku usando EasyOCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    # Redimensiona para tamanho fixo (divisível por n)
    size = 450
    thresh = cv2.resize(thresh, (size, size))

    # Divide a imagem em células n x n
    cells = np.vsplit(thresh, n)
    board = []

    for row in cells:
        row_cells = np.hsplit(row, n)
        linha = []
        for cell in row_cells:
            h, w = cell.shape
            margin = 4
            cell = cell[margin:h-margin, margin:w-margin]

            # Reconhecimento com EasyOCR
            result = reader.readtext(cell, detail=0, paragraph=False)
            if result and result[0].isdigit():
                linha.append(int(result[0]))
            else:
                linha.append(0)
        board.append(linha)
    return board

# -------------------------------
# Interface Streamlit
# -------------------------------
if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)
    st.image(img, caption="Imagem Original", use_container_width=True)

    tamanho = st.selectbox("Tamanho do Sudoku:", [3, 9], index=1)

    if st.button("Resolver Sudoku"):
        try:
            with st.spinner("Lendo imagem e reconhecendo números com EasyOCR..."):
                board = ocr_sudoku_easyocr(img, tamanho)
                st.write("Sudoku Detectado:")
                st.table(board)

            if solve_sudoku(board):
                st.success("Sudoku Resolvido com sucesso!")
                st.table(board)
            else:
                st.error("Não foi possível resolver o Sudoku detectado.")
        except Exception as e:
            st.error(f"Erro durante o processamento: {e}")
