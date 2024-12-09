from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3 as sql
import base64
from httpResponses.recognition import RecognitionHttp
from utilsPersonal.save_img import writeImg
from recognition import IAReconocimiento
from hybrid import SistemaRecomendacionHibrido


app = FastAPI()
recog = IAReconocimiento()
system_recomendation = SistemaRecomendacionHibrido()

origins = [
    "http://localhost:5173",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/recognition")
def recognition(item: RecognitionHttp):
    writeImg('img/img.jpg', base64.b64decode(item.getImg().encode('utf-8')))
    name = recog.recognition('img/img.jpg')
    books = system_recomendation.get_recomendations(name)
    con = sql.connect("recomen.db")
    cursor = con.cursor()
    book_res = []
    for book in books:
        res = cursor.execute(
            'SELECT titulo, autor, editorial, genero, link2 FROM libros WHERE titulo = "'
            + book[0]
            + '"'
        )
        for row in res:
            book = {
                "title": row[0],
                "author": row[1],
                "publisher": row[2],
                "genre": row[3],
                "link": row[4],
            }
            book_res.append(book)
    
    user = cursor.execute( 'SELECT * FROM usuarios WHERE codigo = "'+name+'" ')
    for row in user:
        user = {
            "code": row[0],
            "name": row[1],
            "email": row[2],
        }
    cursor.close()
    con.close()
    
    return {"user": user, "books": book_res}
