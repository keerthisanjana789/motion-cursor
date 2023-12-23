import streamlit as st
import cv2
import mediapipe
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        image_height, image_width, _ = image.shape
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_hands = capture_hands.process(rgb_image)
        all_hands = output_hands.multi_hand_landmarks

        if all_hands:
            for hand in all_hands:
                drawing_option.draw_landmarks(image, hand)
                one_hand_landmarks = hand.landmark
                for id, lm in enumerate(one_hand_landmarks):
                    x = int(lm.x * image_width)
                    y = int(lm.y * image_height)

                    if id == 4:
                        cv2.circle(image, (x, y), 10, (0, 255, 255))

        return image

def app():
    st.title("Motion Cursor App")
    st.markdown("### Welcome to Motion Cursor App!")

    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

if __name__ == "__main__":
    app()


