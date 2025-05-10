from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.tool.predictor import Predictor
import numpy as np
import editdistance
import os
from tqdm import tqdm

# chọn vgg_transformer hoặc vgg_seq2seq 
config = Cfg.load_config_from_name('vgg_transformer')
config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '

dataset_params = {
    'name':'hw', 
    'data_root':'/root/OCR/converted_data', # thư mục chứa dữ liệu bao gồm ảnh và nhãn
    'train_annotation':'train_annotation.txt', # ảnh và nhãn tập train
    'valid_annotation':'val_annotation.txt' # ảnh và nhãn tập test
}

params = {
         'print_every':200, # hiển thị loss mỗi 200 iteration 
         'valid_every':10000, # đánh giá độ chính xác mô hình mỗi 10000 iteraction
          'iters':40000, # Huấn luyện 30000 lần
          'export':'./weights/transformerocr.pth', # lưu model được huấn luyện tại này
          'metrics': 10000 # sử dụng 10000 ảnh của tập test để đánh giá mô hình
         }

# update custom config của các bạn
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0' # device để huấn luyện mô hình, để sử dụng cpu huấn luyện thì thay bằng 'cpu'
# config['weights'] = '/root/OCR/weights/transformerocr.pth'
trainer = Trainer(config, pretrained=True)

# sử dụng lệnh này để visualize tập train, bao gồm cả augmentation 
trainer.visualize_dataset()

# bắt đầu huấn luyện 
trainer.train()

# visualize kết quả dự đoán của mô hình
trainer.visualize_prediction()

# huấn luyện xong thì nhớ lưu lại config để dùng cho Predictor
trainer.config.save('config.yml')

# Hàm tính Character Error Rate (CER)
def compute_cer(pred_text, gt_text):
    dist = editdistance.eval(pred_text, gt_text)
    return dist / max(len(gt_text), 1)

# Hàm tính Word Error Rate (WER)
def compute_wer(pred_text, gt_text):
    pred_words = pred_text.split()
    gt_words = gt_text.split()
    dist = editdistance.eval(pred_words, gt_words)
    return dist / max(len(gt_words), 1)

# Hàm kiểm tra có lỗi trong dự đoán so với ground truth (dùng cho SER)
def has_error(pred_text, gt_text):
    return pred_text != gt_text

# Đánh giá mô hình với CER, WER và SER
def evaluate_model(config, model_path):
    print("Evaluating model with CER, WER, and SER metrics...")
    
    # Tạo predictor từ config và model đã huấn luyện
    predictor = Predictor(config)
    
    # Đọc file annotation của tập validation
    val_path = os.path.join(config['dataset']['data_root'], config['dataset']['valid_annotation'])
    with open(val_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_cer = 0.0
    total_wer = 0.0
    error_sequences = 0  # Số lượng sequences có ít nhất 1 lỗi (dùng cho SER)
    count = 0
    
    # Lưu kết quả chi tiết
    detail_results = []
    
    # Giới hạn số lượng mẫu đánh giá (tùy chọn)
    eval_samples = min(len(lines), 1000)  # Đánh giá tối đa 1000 mẫu
    
    for line in tqdm(lines[:eval_samples], desc="Evaluating"):
        line = line.strip()
        if not line:
            continue
            
        img_path, gt_text = line.split('\t')
        img_path = os.path.join(config['dataset']['data_root'], img_path)
        
        # Dự đoán text từ ảnh
        try:
            pred_text = predictor.predict(img_path)
            
            # Tính CER và WER
            cer = compute_cer(pred_text, gt_text)
            wer = compute_wer(pred_text, gt_text)
            
            # Kiểm tra có lỗi trong sequence hay không
            if has_error(pred_text, gt_text):
                error_sequences += 1
            
            total_cer += cer
            total_wer += wer
            count += 1
            
            # Lưu kết quả chi tiết
            detail_results.append({
                'img_path': img_path,
                'gt_text': gt_text,
                'pred_text': pred_text,
                'cer': cer,
                'wer': wer,
                'has_error': has_error(pred_text, gt_text)
            })
            
            # In kết quả cho một số mẫu
            if count % 100 == 0:
                print(f"Sample {count}:")
                print(f"Ground truth: {gt_text}")
                print(f"Prediction: {pred_text}")
                print(f"CER: {cer:.4f}, WER: {wer:.4f}, Has Error: {has_error(pred_text, gt_text)}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Tính trung bình CER, WER và SER
    avg_cer = total_cer / count if count > 0 else 0
    avg_wer = total_wer / count if count > 0 else 0
    ser = error_sequences / count if count > 0 else 0  # Sequence Error Rate
    
    print(f"\nEvaluation Results:")
    print(f"Number of samples evaluated: {count}")
    print(f"Average Character Error Rate (CER): {avg_cer:.4f}")
    print(f"Average Word Error Rate (WER): {avg_wer:.4f}")
    print(f"Sequence Error Rate (SER): {ser:.4f}")
    
    return avg_cer, avg_wer, ser, detail_results

# Sau khi huấn luyện xong, đánh giá mô hình
model_path = params['export']
cer, wer, ser, detail_results = evaluate_model(config, model_path)

# Lưu kết quả đánh giá vào file
with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Model: {model_path}\n")
    f.write(f"Average Character Error Rate (CER): {cer:.4f}\n")
    f.write(f"Average Word Error Rate (WER): {wer:.4f}\n")
    f.write(f"Sequence Error Rate (SER): {ser:.4f}\n")

# Lưu kết quả chi tiết vào file CSV (tùy chọn)
import csv
with open('detailed_evaluation_results.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['img_path', 'gt_text', 'pred_text', 'cer', 'wer', 'has_error']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for result in detail_results:
        writer.writerow(result)

print("Evaluation complete. Results saved to 'evaluation_results.txt' and 'detailed_evaluation_results.csv'")