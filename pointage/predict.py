def use():
    image = cv2.imread(PATH_IMAGE)
    all_faces, box_coor = extract_face(image)
    for i, face in enumerate(all_faces):
        moyenne, variance = face.mean(), face.std()
        face = (face - moyenne) / variance
        face = face.reshape(1, 160, 160, 3)
        prediction = MODEL_FACENET.predict(face)
        prediciton_face = .predict(prediction)
        prediciton_face_proba = model_svm_poly.predict_proba(prediction)
        aa, bb = np.argmax(prediciton_face_proba), np.max(prediciton_face_proba)
        if bb > THRESHOLD:
            x, y, w, h = box_coor[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.putText(image, out_encoder.inverse_transform(np.array(prediciton_face))[0], (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 1)

    plt.figure(figsize=(100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
