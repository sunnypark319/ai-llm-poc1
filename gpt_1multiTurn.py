from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.getenv('OPENAI_API_KEY')

client=OpenAI(api_key=api_key)

messages = [
    {"role": "system", "content": "너는 배트맨에 나오는 조커이며, 그 캐릭터에 부합하게 답변해줘"}
]

while True:
    user_input = input("사용자:")
    if user_input.lower() == "exit":
        break

    # 대화 내역에 사용자 메시지 추가
    messages.append({"role": "user", "content": user_input})

    # OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=messages
    )

    # AI 응답 저장 및 출력
    ai_message = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ai_message})
    print("AI:", ai_message)