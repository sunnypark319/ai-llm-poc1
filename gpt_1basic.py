from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')

client=OpenAI(api_key=api_key)

response=client.chat.completions.create(
    model="gpt-4o",
    temperature=0.1,
    messages=[
        {"role":"system","content": "너는 배트맨에 나오는 조커야이며, 그  캐릭터에 부합하게 답변해줘"},
        {"role":"user","content": "세상에서 누가 제일 예쁘니?"},
    ]
)

print(response)

print('----')

print(response.choices[0].message.content)