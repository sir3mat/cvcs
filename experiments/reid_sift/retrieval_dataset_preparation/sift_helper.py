from cv2 import dnn_superres

class SIFTHelper():

    # super resolution on image
    def super_res(self, img, path):
        sr = dnn_superres.DnnSuperResImpl_create()
        
        sr.readModel(path)

        sr.setModel("edsr", 2)

        result = sr.upsample(img)
        return result
