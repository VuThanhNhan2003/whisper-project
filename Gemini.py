import json
import os

from dotenv import load_dotenv
from google.generativeai import GenerativeModel

# Khởi tạo model
import google.generativeai as genai


def process_text(input, model_name:str)-> None:
    """
			Hàm xử lý các segment .
		  Args:
            input (list): Danh sách các segment cần xử lý.
			  model_name (str): Tên mô hình.
		  Returns:
			   Văn bản đã xử lý.
	"""
    gemini_model = GenerativeModel(model_name=model_name)
    results=[]
    data = []
    total_tokens = 0
    #2. Duyệt qua từng segmemt và chỉnh sửa
    for item in input[:5]:  # Giới hạn 5 data mẫu để thử
        original_text = item.get('text')
        response, token = edit_text(original_text, gemini_model)
        edited_text, reason = extract_json_from_response(response)
        data.append({
            'id': item.get('id'),
            'start': item.get('start'),
            'end': item.get('end'),
            'origin': item.get('text'),
            'edit': edited_text,
            'reason': reason,
            'token': token
        })
        total_tokens += token
    results.append({
        'data': data,
        'total_tokens': total_tokens,
        'model': "gemma-3n-e2b-it"
    })
    # 3. Lưu kết quả vào file JSON mới
    with open('output_edited.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
def edit_text(text,model)-> tuple[str,int]:
    """
        Sử dụng mô hình Gemini để chỉnh sửa lỗi chính tả và ngữ pháp trong văn bản tiếng Việt .
      Args:
          text (str): The input text to be edited.
      Returns:
          response (str): Kết quả xử lý.
            token (int): Số token đã sử dụng.
      """
    prompt = (f":Bạn là một công cụ sửa lỗi chính tả và ngữ pháp trong tiếng Việt."
              f"\n Nhiệm vụ: 1. Chỉ sửa lỗi chính tả, dấu câu, hoặc lỗi ngữ pháp."
              f"\n2. Không thay đổi ý nghĩa gốc của câu."
              f" \n3. Trả về kết quả dưới dạng JSON với 2 trường: - text: câu sau khi chỉnh sửa. - reason: giải thích ngắn gọn các lỗi đã sửa."
              f"\n Đầu vào:\n{text}.")
    try:
        response = model.generate_content(prompt)
        return response.text.strip(), response.usage_metadata.total_token_count
    except Exception as e:
        print(f"Error generating content: {e}")
        return text  # Return the original text as a fallback
def extract_json_from_response(response: str) -> (str, str):
    """
        Hàm trích xuất dữ liệu JSON từ phản hồi của mô hình Gemini.
      Args:
          response (str): Kết quả phản hồi JSON-formatted .
      Returns:
            text (str): Văn bản đã chỉnh sửa.
            reason (str): Lý do chỉnh sửa.
      """
    json_str = response.replace('```json', '').replace('```', '').strip()
    try:
        data = json.loads(json_str)
        # Trích xuất dữ liệu
        return data['text'], data['reason']
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return "", ""
if __name__ == '__main__':
    load_dotenv()  # Loads variables from .env into environment
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    #1. Đọc file JSON đã được xử lý bởi Whisper
    with open('output.json', 'r', encoding='utf-8') as f:
        input = json.load(f)  # data is now a list of dicts (array of objects)
    model="gemma-3n-e2b-it"
    process_text(input,model)



