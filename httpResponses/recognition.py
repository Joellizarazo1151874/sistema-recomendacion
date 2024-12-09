from pydantic import BaseModel

class RecognitionHttp(BaseModel):
    img: str

    def getImg(self):
        return self.img