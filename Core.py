import time
import serial

cap = cv2.VideoCapture('http://admin:pw2@ipaddress/video.cgi?.mjpg')
ser = serial.Serial('/dev/cu.usbmodem14201', 9800, timeout=1)
time.sleep(2)
ser.write(b'H')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc(*'jpeg')
#out = cv2.VideoWriter('output2.mov', fourcc, 20.0, (width, height))

start_time = time.time()
timer = 2 * 50 

prey_detected = False
run_once = False

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.85,
                agnostic_mode=False)

  

    if 1 in detections['detection_classes'] and np.max(detections['detection_scores'][detections['detection_classes'] == 1]) >= 0.95 and not prey_detected:
        ser.write(b'L')
        locked = 1
        start_time = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'jpeg')
        out = cv2.VideoWriter('output2_' + str(int(time.time())) + '.mov', fourcc, 20.0, (width, height))
        prey_detected = True
        
    if prey_detected:
        elapsed_time = int(time.time() - start_time)
        remaining_time = timer - elapsed_time
        minutes, seconds = divmod(remaining_time, 60)
        
        cv2.putText(image_np_with_detections,
                    "Mouse detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(image_np_with_detections,
                    "Door locked for {:02d}:{:02d}min".format(minutes, seconds),
                    (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        out.write(image_np_with_detections)
        if remaining_time<=0:
            start_time = time.time()
            timer = 2 * 60 
            locked = 0
            ser.write(b'H')
            out.release()
            prey_detected = False
            
        if not run_once:
            #ser.write(b'H')
            run_once = True
            
            
    if not prey_detected:
        cv2.putText(image_np_with_detections,
                    "Door unlocked for Luna",
                    (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        
    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    #out.write(image_np_with_detections)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
