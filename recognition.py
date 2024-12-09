import cv2
import numpy as np
import face_recognition as fr
import os

class IAReconocimiento():

  # Función para encontrar la codificación de las imágenes.
  def findEncodings(self,images):
      encodeList = []
      for img in images:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          encode = fr.face_encodings(img)[0]
          encodeList.append(encode)
      return encodeList

  def recognition(self,filename):
        # Cargue el img desde el directorio de la base de datos.
        path = "imgs"
        images = []
        classNames = []
        myList = os.listdir(path)

        # Recorre las imágenes y agrégalas a la lista.
        for cl in myList:
            curImg = cv2.imread(f"{path}/{cl}")
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        # Llame a la función para encontrar las codificaciones de las imágenes.
        encodeListKnown = self.findEncodings(images)
        img = cv2.imread(filename)
        result_name = self.findFace(img, encodeListKnown, classNames)

        return result_name  # Devuelve el resultado en lugar de imprimirlo
# Función para encontrar la cara.
  def findFace(self,img, encodeListKnown,classNames):
    #   imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    #   imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

      # Encuentra las ubicaciones de las caras
      facesCurFrame = fr.face_locations(img)
      encodesCurFrame = fr.face_encodings(img, facesCurFrame)
      # Recorre las ubicaciones y codificaciones de las caras.
      for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
          matches = fr.compare_faces(encodeListKnown, encodeFace)
          faceDis = fr.face_distance(encodeListKnown, encodeFace)

          # Encuentra el índice de la distancia mínima.
          matchIndex = np.argmin(faceDis)

          # Verifica si la coincidencia es mayor o igual al 95%
          if matches[matchIndex] and faceDis[matchIndex] < 0.95:
              name = classNames[matchIndex].upper()
          else:
              name = "Desconocido"

          return name


