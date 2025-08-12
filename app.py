import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ THÊM
import numpy as np
import requests

app = Flask(__name__)
CORS(app)  # ✅ CHO PHÉP MỌI ORIGIN TRUY CẬP API

class SlideSolver:
    def __init__(self, puzzle_piece: bytes, background: bytes):
        self.background = self._read_image(background)
        self.puzzle_piece = self._read_image(puzzle_piece)

    @staticmethod
    def _read_image(image_source):
        return cv2.imdecode(np.frombuffer(image_source, np.uint8), cv2.IMREAD_UNCHANGED)

    def find_puzzle_piece_position(self):
        # Remove alpha channel if needed
        if self.puzzle_piece.shape[2] == 4:
            alpha = self.puzzle_piece[:, :, 3]
            _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
            self.puzzle_piece = cv2.bitwise_and(
                self.puzzle_piece[:, :, :3],
                self.puzzle_piece[:, :, :3],
                mask=mask
            )

        # Convert both to edge maps
        edge_puzzle_piece = cv2.Canny(self.puzzle_piece, 100, 200)
        edge_background = cv2.Canny(self.background, 100, 200)

        # Convert to 3 channels
        edge_puzzle_piece_rgb = cv2.cvtColor(edge_puzzle_piece, cv2.COLOR_GRAY2RGB)
        edge_background_rgb = cv2.cvtColor(edge_background, cv2.COLOR_GRAY2RGB)

        # Match template
        res = cv2.matchTemplate(edge_background_rgb, edge_puzzle_piece_rgb, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        w = edge_puzzle_piece.shape[1]
        center_x = top_left[0] + w // 2

        return center_x - 41

def get_image_bytes(url):
    return requests.get(url).content

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    bg_url = data["bg_url"]
    puzzle_url = data["slice_url"]

    background = get_image_bytes(bg_url)
    puzzle_piece = get_image_bytes(puzzle_url)

    solver = SlideSolver(puzzle_piece, background)
    distance = solver.find_puzzle_piece_position()
    return jsonify({"distance": distance})
