import cv2


def draw_highlighted_text(img, text, position, text_color, bg_color):
    font_scale, thickness = 0.7, 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - text_size[1] - 5), (x + text_size[0] + 4, y + 5), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x + 2, y), font, font_scale, text_color, thickness)
