import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2


model_path = "models/inswapper_128.onnx"
input_picture = cv2.imread("data/photos-for-ai/input/steve.png")
destination_picture = cv2.imread(
    # "data/photos-for-ai/destination/Hubert_Humphrey_Portrait_Colorized.jpg"
    # "data/photos-for-ai/destination/Bronzino_-_Portrait_of_a_Young_Man,_1550-1555.jpg"
    # "data/photos-for-ai/destination/preview16.jpg"
    # Beckham
    # "data/more-samples/0e09ee6468623715aec8be59d0c1dab8.jpg"
    "data/more-samples/upbeat-homeless-man.jpg"
)

value = 0
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(model_path, download=True, download_zip=True)


def swap_faces(faceSource, sourceFaceId, faceDestination, destFaceId):
    faces = app.get(faceSource)
    faces = sorted(faces, key=lambda x: x.bbox[0])
    if len(faces) < sourceFaceId or sourceFaceId < 1:
        print("we should never get here")
        # print(
        #     f"Source image only contains {len(faces)} faces, but you requested face {sourceFaceId}"
        # )

    source_face = faces[sourceFaceId - 1]

    res_faces = app.get(faceDestination)
    res_faces = sorted(res_faces, key=lambda x: x.bbox[0])
    if len(res_faces) < destFaceId or destFaceId < 1:
        print("we should never get here2")
        # print(
        #     f"Source image only contains {len(faces)} faces, but you requested face {sourceFaceId}"
        # )
    res_face = res_faces[destFaceId - 1]

    result = swapper.get(faceDestination, res_face, source_face, paste_back=True)

    global value
    value = value + 1
    print(f"processed: {value}...")

    for face in faces:
        res = swapper.get(result, face, source_face, paste_back=True)
    cv2.imwrite("./t1_swapped.jpg", res)
    # return result


swap_faces(
    faceSource=input_picture,
    sourceFaceId=1,
    faceDestination=destination_picture,
    destFaceId=1,
)

print("Done")
