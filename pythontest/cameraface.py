import sys
sys.path.append('d:/pythontest/detectface')
import tensorflow as tf
import detect_face
import cv2

minsize = 20 # minimum size of face 
threshold = [ 0.6, 0.7, 0.7 ] # three steps's threshold 
factor = 0.709 # scale factor 
gpu_memory_fraction=1.0

print('Creating networks and loading parameters')

with tf.Graph().as_default(): 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) 
    sess = tf.Session() 
    with sess.as_default(): 
      pnet, rnet, onet = detect_face.create_mtcnn(sess, None)



camera_number = 0
cap = cv2.VideoCapture(camera_number)

while(True):
    ret,frame = cap.read()
    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor) 
    nrof_faces = bounding_boxes.shape[0]#人脸数目 
    print('找到人脸数目为：{}'.format(nrof_faces))
    crop_faces=[] 
    for face_position in bounding_boxes: 
        face_position=face_position.astype(int) 
        print(face_position[0:4]) 
        cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)     
    cv2.imshow('frame', frame)
    if cv2.waitKey(10)== ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
