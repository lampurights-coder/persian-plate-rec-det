
# import cv2
# import numpy as np
# import os
# import re
# import paddle
# import math
# from PIL import Image
# import time
# import sys

# def create_predictor(args, mode, logger):
#     print(args, mode, logger)
#     if mode == "rec":
#         model_dir = args.rec_model_dir
#     if model_dir is None:
#         logger.info("not find {} model file path {}".format(mode, model_dir))
#         sys.exit(0)
#     if args.use_onnx:
#         import onnxruntime as ort
#         model_file_path = model_dir
#         if not os.path.exists(model_file_path):
#             raise ValueError("not find model file path {}".format(model_file_path))
#         sess_options = args.onnx_sess_options or None
#         if args.onnx_providers and len(args.onnx_providers) > 0:
#             sess = ort.InferenceSession(
#                 model_file_path,
#                 providers=args.onnx_providers,
#                 sess_options=sess_options,
#             )
#         elif args.use_gpu:
#             sess = ort.InferenceSession(
#                 model_file_path,
#                 providers=[
#                     (
#                         "CUDAExecutionProvider",
#                         {"device_id": args.gpu_id, "cudnn_conv_algo_search": "DEFAULT"},
#                     )
#                 ],
#                 sess_options=sess_options,
#             )
#         else:
#             sess = ort.InferenceSession(
#                 model_file_path,
#                 providers=["CPUExecutionProvider"],
#                 sess_options=sess_options,
#             )
#         inputs = sess.get_inputs()
#         return (
#             sess,
#             inputs[0] if len(inputs) == 1 else [vo.name for vo in inputs],
#             None,
#             None,
#         )

# def check_and_read(img_path):
#     if os.path.basename(img_path)[-3:].lower() == "gif":
#         gif = cv2.VideoCapture(img_path)
#         ret, frame = gif.read()
#         if not ret:
#             return None, False
#         if len(frame.shape) == 2 or frame.shape[-1] == 1:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         imgvalue = frame[:, :, ::-1]
#         return imgvalue, True, False
#     elif os.path.basename(img_path)[-3:].lower() == "pdf":
#         from paddle.utils import try_import

#         fitz = try_import("fitz")
#         from PIL import Image

#         imgs = []
#         with fitz.open(img_path) as pdf:
#             for pg in range(0, pdf.page_count):
#                 page = pdf[pg]
#                 mat = fitz.Matrix(2, 2)
#                 pm = page.get_pixmap(matrix=mat, alpha=False)

#                 # if width or height > 2000 pixels, don't enlarge the image
#                 if pm.width > 2000 or pm.height > 2000:
#                     pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

#                 img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
#                 img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#                 imgs.append(img)
#             return imgs, False, True
#     return None, False, False


# class BaseRecLabelDecode(object):
#     """Convert between text-label and text-index"""

#     def __init__(self, character_dict_path=None, use_space_char=False):
#         self.beg_str = "sos"
#         self.end_str = "eos"
#         self.reverse = False
#         self.character_str = []

#         if character_dict_path is None:
#             self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
#             dict_character = list(self.character_str)
#         else:
#             with open(character_dict_path, "rb") as fin:
#                 lines = fin.readlines()
#                 for line in lines:
#                     line = line.decode("utf-8").strip("\n").strip("\r\n")
#                     self.character_str.append(line)
#             if use_space_char:
#                 self.character_str.append(" ")
#             dict_character = list(self.character_str)
#             if "arabic" in character_dict_path:
#                 self.reverse = True

#         dict_character = self.add_special_char(dict_character)
#         self.dict = {}
#         for i, char in enumerate(dict_character):
#             self.dict[char] = i
#         self.character = dict_character

#     def pred_reverse(self, pred):
#         pred_re = []
#         c_current = ""
#         for c in pred:
#             if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
#                 if c_current != "":
#                     pred_re.append(c_current)
#                 pred_re.append(c)
#                 c_current = ""
#             else:
#                 c_current += c
#         if c_current != "":
#             pred_re.append(c_current)

#         return "".join(pred_re[::-1])

#     def add_special_char(self, dict_character):
#         return dict_character

#     def get_word_info(self, text, selection):
#         """
#         Group the decoded characters and record the corresponding decoded positions.

#         Args:
#             text: the decoded text
#             selection: the bool array that identifies which columns of features are decoded as non-separated characters
#         Returns:
#             word_list: list of the grouped words
#             word_col_list: list of decoding positions corresponding to each character in the grouped word
#             state_list: list of marker to identify the type of grouping words, including two types of grouping words:
#                         - 'cn': continuous chinese characters (e.g., 你好啊)
#                         - 'en&num': continuous english characters (e.g., hello), number (e.g., 123, 1.123), or mixed of them connected by '-' (e.g., VGG-16)
#                         The remaining characters in text are treated as separators between groups (e.g., space, '(', ')', etc.).
#         """
#         state = None
#         word_content = []
#         word_col_content = []
#         word_list = []
#         word_col_list = []
#         state_list = []
#         valid_col = np.where(selection == True)[0]

#         for c_i, char in enumerate(text):
#             if "\u4e00" <= char <= "\u9fff":
#                 c_state = "cn"
#             elif bool(re.search("[a-zA-Z0-9]", char)):
#                 c_state = "en&num"
#             else:
#                 c_state = "splitter"

#             if (
#                 char == "."
#                 and state == "en&num"
#                 and c_i + 1 < len(text)
#                 and bool(re.search("[0-9]", text[c_i + 1]))
#             ):  # grouping floating number
#                 c_state = "en&num"
#             if (
#                 char == "-" and state == "en&num"
#             ):  # grouping word with '-', such as 'state-of-the-art'
#                 c_state = "en&num"

#             if state == None:
#                 state = c_state

#             if state != c_state:
#                 if len(word_content) != 0:
#                     word_list.append(word_content)
#                     word_col_list.append(word_col_content)
#                     state_list.append(state)
#                     word_content = []
#                     word_col_content = []
#                 state = c_state

#             if state != "splitter":
#                 word_content.append(char)
#                 word_col_content.append(valid_col[c_i])

#         if len(word_content) != 0:
#             word_list.append(word_content)
#             word_col_list.append(word_col_content)
#             state_list.append(state)

#         return word_list, word_col_list, state_list

#     def decode(
#         self,
#         text_index,
#         text_prob=None,
#         is_remove_duplicate=False,
#         return_word_box=False,
#     ):
#         """convert text-index into text-label."""
#         result_list = []
#         ignored_tokens = self.get_ignored_tokens()
#         batch_size = len(text_index)
#         for batch_idx in range(batch_size):
#             selection = np.ones(len(text_index[batch_idx]), dtype=bool)
#             if is_remove_duplicate:
#                 selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
#             for ignored_token in ignored_tokens:
#                 selection &= text_index[batch_idx] != ignored_token

#             char_list = [
#                 self.character[text_id] for text_id in text_index[batch_idx][selection]
#             ]
#             if text_prob is not None:
#                 conf_list = text_prob[batch_idx][selection]
#             else:
#                 conf_list = [1] * len(selection)
#             if len(conf_list) == 0:
#                 conf_list = [0]

#             text = "".join(char_list)

#             if self.reverse:  # for arabic rec
#                 text = self.pred_reverse(text)

#             if return_word_box:
#                 word_list, word_col_list, state_list = self.get_word_info(
#                     text, selection
#                 )
#                 result_list.append(
#                     (
#                         text,
#                         np.mean(conf_list).tolist(),
#                         [
#                             len(text_index[batch_idx]),
#                             word_list,
#                             word_col_list,
#                             state_list,
#                         ],
#                     )
#                 )
#             else:
#                 result_list.append((text, np.mean(conf_list).tolist()))
#         return result_list

#     def get_ignored_tokens(self):
#         return [0]  # for ctc blank


# class SARLabelDecode(BaseRecLabelDecode):
#     """Convert between text-label and text-index"""

#     def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
#         super(SARLabelDecode, self).__init__(character_dict_path, use_space_char)

#         self.rm_symbol = kwargs.get("rm_symbol", False)

#     def add_special_char(self, dict_character):
#         beg_end_str = "<BOS/EOS>"
#         unknown_str = "<UKN>"
#         padding_str = "<PAD>"
#         dict_character = dict_character + [unknown_str]
#         self.unknown_idx = len(dict_character) - 1
#         dict_character = dict_character + [beg_end_str]
#         self.start_idx = len(dict_character) - 1
#         self.end_idx = len(dict_character) - 1
#         dict_character = dict_character + [padding_str]
#         self.padding_idx = len(dict_character) - 1
#         return dict_character

#     def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
#         """convert text-index into text-label."""
#         result_list = []
#         ignored_tokens = self.get_ignored_tokens()

#         batch_size = len(text_index)
#         for batch_idx in range(batch_size):
#             char_list = []
#             conf_list = []
#             for idx in range(len(text_index[batch_idx])):
#                 if text_index[batch_idx][idx] in ignored_tokens:
#                     continue
#                 if int(text_index[batch_idx][idx]) == int(self.end_idx):
#                     if text_prob is None and idx == 0:
#                         continue
#                     else:
#                         break
#                 if is_remove_duplicate:
#                     # only for predict
#                     if (
#                         idx > 0
#                         and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
#                     ):
#                         continue
#                 char_list.append(self.character[int(text_index[batch_idx][idx])])
#                 if text_prob is not None:
#                     conf_list.append(text_prob[batch_idx][idx])
#                 else:
#                     conf_list.append(1)
#             text = "".join(char_list)
#             if self.rm_symbol:
#                 comp = re.compile("[^A-Z^a-z^0-9^\u4e00-\u9fa5]")
#                 text = text.lower()
#                 text = comp.sub("", text)
#             result_list.append((text, np.mean(conf_list).tolist()))
#         return result_list

#     def __call__(self, preds, label=None, *args, **kwargs):
#         if isinstance(preds, paddle.Tensor):
#             preds = preds.numpy()
#         preds_idx = preds.argmax(axis=2)
#         preds_prob = preds.max(axis=2)

#         text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)

#         if label is None:
#             return text
#         label = self.decode(label, is_remove_duplicate=False)
#         return text, label

#     def get_ignored_tokens(self):
#         return [self.padding_idx]


# class TextRecognizer(object):
#     def __init__(self, 
#                  rec_image_shape,
#                  rec_char_dict_path,
#                  ):
            
#             self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
#             self.rec_batch_num = 6
#             self.rec_algorithm = 'RobustScanner'

#             postprocess_params = {
#                 "name": "SARLabelDecode",
#                 "character_dict_path": rec_char_dict_path,
#                 "use_space_char": False,
#                 "rm_symbol": True,
#             }
#             self.postprocess_op = SARLabelDecode(**postprocess_params)
#             self.return_word_box = False
#             self.use_onnx = True

#     def resize_norm_img(self, img, max_wh_ratio):
#         imgC, imgH, imgW = self.rec_image_shape
#         if self.rec_algorithm == "NRTR" or self.rec_algorithm == "ViTSTR":
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             image_pil = Image.fromarray(np.uint8(img))
#             if self.rec_algorithm == "ViTSTR":
#                 img = image_pil.resize([imgW, imgH], Image.BICUBIC)
#             else:
#                 img = image_pil.resize([imgW, imgH], Image.Resampling.LANCZOS)
#             img = np.array(img)
#             norm_img = np.expand_dims(img, -1)
#             norm_img = norm_img.transpose((2, 0, 1))
#             if self.rec_algorithm == "ViTSTR":
#                 norm_img = norm_img.astype(np.float32) / 255.0
#             else:
#                 norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
#             return norm_img
#         elif self.rec_algorithm == "RFL":
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
#             resized_image = resized_image.astype("float32")
#             resized_image = resized_image / 255
#             resized_image = resized_image[np.newaxis, :]
#             resized_image -= 0.5
#             resized_image /= 0.5
#             return resized_image

#         assert imgC == img.shape[2]
#         imgW = int((imgH * max_wh_ratio))
#         if self.use_onnx:
#             w = self.input_tensor.shape[3:][0]
#             if isinstance(w, str):
#                 pass
#             elif w is not None and w > 0:
#                 imgW = w
#         h, w = img.shape[:2]
#         ratio = w / float(h)
#         if math.ceil(imgH * ratio) > imgW:
#             resized_w = imgW
#         else:
#             resized_w = int(math.ceil(imgH * ratio))
#         if self.rec_algorithm == "RARE":
#             if resized_w > self.rec_image_shape[2]:
#                 resized_w = self.rec_image_shape[2]
#             imgW = self.rec_image_shape[2]
#         resized_image = cv2.resize(img, (resized_w, imgH))
#         resized_image = resized_image.astype("float32")
#         resized_image = resized_image.transpose((2, 0, 1)) / 255
#         resized_image -= 0.5
#         resized_image /= 0.5
#         padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
#         padding_im[:, :, 0:resized_w] = resized_image
#         return padding_im

#     def resize_norm_img_vl(self, img, image_shape):
#         imgC, imgH, imgW = image_shape
#         img = img[:, :, ::-1]
#         resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
#         resized_image = resized_image.astype("float32")
#         resized_image = resized_image.transpose((2, 0, 1)) / 255
#         return resized_image

#     def resize_norm_img_srn(self, img, image_shape):
#         imgC, imgH, imgW = image_shape
#         img_black = np.zeros((imgH, imgW))
#         im_hei = img.shape[0]
#         im_wid = img.shape[1]
#         if im_wid <= im_hei * 1:
#             img_new = cv2.resize(img, (imgH * 1, imgH))
#         elif im_wid <= im_hei * 2:
#             img_new = cv2.resize(img, (imgH * 2, imgH))
#         elif im_wid <= im_hei * 3:
#             img_new = cv2.resize(img, (imgH * 3, imgH))
#         else:
#             img_new = cv2.resize(img, (imgW, imgH))
#         img_np = np.asarray(img_new)
#         img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#         img_black[:, 0 : img_np.shape[1]] = img_np
#         img_black = img_black[:, :, np.newaxis]
#         row, col, c = img_black.shape
#         c = 1
#         return np.reshape(img_black, (c, row, col)).astype(np.float32)

#     def srn_other_inputs(self, image_shape, num_heads, max_text_length):
#         imgC, imgH, imgW = image_shape
#         feature_dim = int((imgH / 8) * (imgW / 8))
#         encoder_word_pos = np.array(range(0, feature_dim)).reshape((feature_dim, 1)).astype("int64")
#         gsrm_word_pos = np.array(range(0, max_text_length)).reshape((max_text_length, 1)).astype("int64")
#         gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
#         gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape([-1, 1, max_text_length, max_text_length])
#         gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1, [1, num_heads, 1, 1]).astype("float32") * [-1e9]
#         gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape([-1, 1, max_text_length, max_text_length])
#         gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2, [1, num_heads, 1, 1]).astype("float32") * [-1e9]
#         encoder_word_pos = encoder_word_pos[np.newaxis, :]
#         gsrm_word_pos = gsrm_word_pos[np.newaxis, :]
#         return [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2]

#     def process_image_srn(self, img, image_shape, num_heads, max_text_length):
#         norm_img = self.resize_norm_img_srn(img, image_shape)
#         norm_img = norm_img[np.newaxis, :]
#         [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = self.srn_other_inputs(image_shape, num_heads, max_text_length)
#         gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
#         gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
#         encoder_word_pos = encoder_word_pos.astype(np.int64)
#         gsrm_word_pos = gsrm_word_pos.astype(np.int64)
#         return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2)

#     def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
#         imgC, imgH, imgW_min, imgW_max = image_shape
#         h = img.shape[0]
#         w = img.shape[1]
#         valid_ratio = 1.0
#         width_divisor = int(1 / width_downsample_ratio)
#         ratio = w / float(h)
#         resize_w = math.ceil(imgH * ratio)
#         if resize_w % width_divisor != 0:
#             resize_w = round(resize_w / width_divisor) * width_divisor
#         if imgW_min is not None:
#             resize_w = max(imgW_min, resize_w)
#         if imgW_max is not None:
#             valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
#             resize_w = min(imgW_max, resize_w)
#         resized_image = cv2.resize(img, (resize_w, imgH))
#         resized_image = resized_image.astype("float32")
#         if image_shape[0] == 1:
#             resized_image = resized_image / 255
#             resized_image = resized_image[np.newaxis, :]
#         else:
#             resized_image = resized_image.transpose((2, 0, 1)) / 255
#         resized_image -= 0.5
#         resized_image /= 0.5
#         resize_shape = resized_image.shape
#         padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
#         padding_im[:, :, 0:resize_w] = resized_image
#         pad_shape = padding_im.shape
#         return padding_im, resize_shape, pad_shape, valid_ratio

#     def resize_norm_img_spin(self, img):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
#         img = np.array(img, np.float32)
#         img = np.expand_dims(img, -1)
#         img = img.transpose((2, 0, 1))
#         mean = [127.5]
#         std = [127.5]
#         mean = np.array(mean, dtype=np.float32)
#         std = np.array(std, dtype=np.float32)
#         mean = np.float32(mean.reshape(1, -1))
#         stdinv = 1 / np.float32(std.reshape(1, -1))
#         img -= mean
#         img *= stdinv
#         return img

#     def resize_norm_img_svtr(self, img, image_shape):
#         imgC, imgH, imgW = image_shape
#         max_wh_ratio = imgW * 1.0 / imgH
#         h, w = img.shape[0], img.shape[1]
#         ratio = w * 1.0 / h
#         max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
#         imgW = int(imgH * max_wh_ratio)
#         if math.ceil(imgH * ratio) > imgW:
#             resized_w = imgW
#         else:
#             resized_w = int(math.ceil(imgH * ratio))
#         resized_image = cv2.resize(img, (resized_w, imgH))
#         resized_image = resized_image.astype("float32")
#         resized_image = resized_image.transpose((2, 0, 1)) / 255
#         resized_image -= 0.5
#         resized_image /= 0.5
#         padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
#         padding_im[:, :, 0:resized_w] = resized_image
#         return padding_im

#     def resize_norm_img_cppd_padding(self, img, image_shape, padding=True, interpolation=cv2.INTER_LINEAR):
#         imgC, imgH, imgW = image_shape
#         h = img.shape[0]
#         w = img.shape[1]
#         if not padding:
#             resized_image = cv2.resize(img, (imgW, imgH), interpolation=interpolation)
#             resized_w = imgW
#         else:
#             ratio = w / float(h)
#             if math.ceil(imgH * ratio) > imgW:
#                 resized_w = imgW
#             else:
#                 resized_w = int(math.ceil(imgH * ratio))
#             resized_image = cv2.resize(img, (resized_w, imgH))
#         resized_image = resized_image.astype("float32")
#         if image_shape[0] == 1:
#             resized_image = resized_image / 255
#             resized_image = resized_image[np.newaxis, :]
#         else:
#             resized_image = resized_image.transpose((2, 0, 1)) / 255
#         resized_image -= 0.5
#         resized_image /= 0.5
#         padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
#         padding_im[:, :, 0:resized_w] = resized_image
#         return padding_im

#     def resize_norm_img_abinet(self, img, image_shape):
#         imgC, imgH, imgW = image_shape
#         resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
#         resized_image = resized_image.astype("float32")
#         resized_image = resized_image / 255.0
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         resized_image = (resized_image - mean[None, None, ...]) / std[None, None, ...]
#         resized_image = resized_image.transpose((2, 0, 1))
#         resized_image = resized_image.astype("float32")
#         return resized_image

#     def norm_img_can(self, img, image_shape):
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         if self.inverse:
#             img = 255 - img
#         if self.rec_image_shape[0] == 1:
#             h, w = img.shape
#             _, imgH, imgW = self.rec_image_shape
#             if h < imgH or w < imgW:
#                 padding_h = max(imgH - h, 0)
#                 padding_w = max(imgW - w, 0)
#                 img_padded = np.pad(img, ((0, padding_h), (0, padding_w)), "constant", constant_values=(255))
#                 img = img_padded
#         img = np.expand_dims(img, 0) / 255.0
#         img = img.astype("float32")
#         return img

#     def pad_(self, img, divable=32):
#         threshold = 128
#         data = np.array(img.convert("LA"))
#         if data[..., -1].var() == 0:
#             data = (data[..., 0]).astype(np.uint8)
#         else:
#             data = (255 - data[..., -1]).astype(np.uint8)
#         data = (data - data.min()) / (data.max() - data.min()) * 255
#         if data.mean() > threshold:
#             gray = 255 * (data < threshold).astype(np.uint8)
#         else:
#             gray = 255 * (data > threshold).astype(np.uint8)
#             data = 255 - data
#         coords = cv2.findNonZero(gray)
#         a, b, w, h = cv2.boundingRect(coords)
#         rect = data[b : b + h, a : a + w]
#         im = Image.fromarray(rect).convert("L")
#         dims = []
#         for x in [w, h]:
#             div, mod = divmod(x, divable)
#             dims.append(divable * (div + (1 if mod > 0 else 0)))
#         padded = Image.new("L", dims, 255)
#         padded.paste(im, (0, 0, im.size[0], im.size[1]))
#         return padded

#     def minmax_size_(self, img, max_dimensions, min_dimensions):
#         if max_dimensions is not None:
#             ratios = [a / b for a, b in zip(img.size, max_dimensions)]
#             if any([r > 1 for r in ratios]):
#                 size = np.array(img.size) // max(ratios)
#                 img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
#         if min_dimensions is not None:
#             padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
#             if padded_size != list(img.size):
#                 padded_im = Image.new("L", padded_size, 255)
#                 padded_im.paste(img, img.getbbox())
#                 img = padded_im
#         return img

#     def norm_img_latexocr(self, img):
#         shape = (1, 1, 3)
#         mean = [0.7931, 0.7931, 0.7931]
#         std = [0.1738, 0.1738, 0.1738]
#         scale = np.float32(1.0 / 255.0)
#         min_dimensions = [32, 32]
#         max_dimensions = [672, 192]
#         mean = np.array(mean).reshape(shape).astype("float32")
#         std = np.array(std).reshape(shape).astype("float32")
#         im_h, im_w = img.shape[:2]
#         if not (min_dimensions[0] <= im_w <= max_dimensions[0] and min_dimensions[1] <= im_h <= max_dimensions[1]):
#             img = Image.fromarray(np.uint8(img))
#             img = self.minmax_size_(self.pad_(img), max_dimensions, min_dimensions)
#             img = np.array(img)
#             im_h, im_w = img.shape[:2]
#             img = np.dstack([img, img, img])
#         img = (img.astype("float32") * scale - mean) / std
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         divide_h = math.ceil(im_h / 16) * 16
#         divide_w = math.ceil(im_w / 16) * 16
#         img = np.pad(img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1))
#         img = img[:, :, np.newaxis].transpose(2, 0, 1)
#         img = img.astype("float32")
#         return img

#     def __call__(self, img_list):
#         img_num = len(img_list)
#         width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
#         indices = np.argsort(np.array(width_list))
#         rec_res = [["", 0.0]] * img_num
#         batch_num = self.rec_batch_num
#         st = time.time()
#         print(batch_num, rec_res, indices, width_list, img_num)

#         for beg_img_no in range(0, img_num, batch_num):
#             end_img_no = min(img_num, beg_img_no + batch_num)
#             norm_img_batch = []
#             if self.rec_algorithm == "SRN":
#                 encoder_word_pos_list = []
#                 gsrm_word_pos_list = []
#                 gsrm_slf_attn_bias1_list = []
#                 gsrm_slf_attn_bias2_list = []
#             if self.rec_algorithm == "SAR":
#                 valid_ratios = []
#             imgC, imgH, imgW = self.rec_image_shape[:3]
#             max_wh_ratio = imgW / imgH
#             wh_ratio_list = []
#             for ino in range(beg_img_no, end_img_no):
#                 h, w = img_list[indices[ino]].shape[0:2]
#                 wh_ratio = w * 1.0 / h
#                 max_wh_ratio = max(max_wh_ratio, wh_ratio)
#                 wh_ratio_list.append(wh_ratio)
                

#             for ino in range(beg_img_no, end_img_no):
#                 if self.rec_algorithm == "SAR":
#                     norm_img, _, _, valid_ratio = self.resize_norm_img_sar(img_list[indices[ino]], self.rec_image_shape)
#                     norm_img = norm_img[np.newaxis, :]
#                     valid_ratio = np.expand_dims(valid_ratio, axis=0)
#                     valid_ratios.append(valid_ratio)
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm == "SRN":
#                     norm_img = self.process_image_srn(img_list[indices[ino]], self.rec_image_shape, 8, 25)
#                     encoder_word_pos_list.append(norm_img[1])
#                     gsrm_word_pos_list.append(norm_img[2])
#                     gsrm_slf_attn_bias1_list.append(norm_img[3])
#                     gsrm_slf_attn_bias2_list.append(norm_img[4])
#                     norm_img_batch.append(norm_img[0])
#                 elif self.rec_algorithm in ["SVTR", "SATRN", "ParseQ", "CPPD"]:
#                     norm_img = self.resize_norm_img_svtr(img_list[indices[ino]], self.rec_image_shape)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm in ["CPPDPadding"]:
#                     norm_img = self.resize_norm_img_cppd_padding(img_list[indices[ino]], self.rec_image_shape)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm in ["VisionLAN", "PREN"]:
#                     norm_img = self.resize_norm_img_vl(img_list[indices[ino]], self.rec_image_shape)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm == "SPIN":
#                     norm_img = self.resize_norm_img_spin(img_list[indices[ino]])
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm == "ABINet":
#                     norm_img = self.resize_norm_img_abinet(img_list[indices[ino]], self.rec_image_shape)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 elif self.rec_algorithm == "RobustScanner":
#                     norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
#                         img_list[indices[ino]],
#                         self.rec_image_shape,
#                         width_downsample_ratio=0.25,
#                     )

#                     norm_img = norm_img[np.newaxis, :]
#                     valid_ratio = np.expand_dims(valid_ratio, axis=0)
#                     valid_ratios = [] if 'valid_ratios' not in locals() else valid_ratios
#                     valid_ratios.append(valid_ratio)
#                     norm_img_batch.append(norm_img)
#                     word_positions_list = [] if 'word_positions_list' not in locals() else word_positions_list
#                     word_positions = np.array(range(0, 40)).astype("int64")
#                     word_positions = np.expand_dims(word_positions, axis=0)
#                     word_positions_list.append(word_positions)
#                 elif self.rec_algorithm == "CAN":
#                     norm_img = self.norm_img_can(img_list[indices[ino]], max_wh_ratio)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                     norm_image_mask = np.ones(norm_img.shape, dtype="float32")
#                     word_label = np.ones([1, 36], dtype="int64")
#                     norm_img_mask_batch = [] if 'norm_img_mask_batch' not in locals() else norm_img_mask_batch
#                     word_label_list = [] if 'word_label_list' not in locals() else word_label_list
#                     norm_img_mask_batch.append(norm_image_mask)
#                     word_label_list.append(word_label)
#                 elif self.rec_algorithm == "LaTeXOCR":
#                     norm_img = self.norm_img_latexocr(img_list[indices[ino]])
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)
#                 else:
#                     norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
#                     norm_img = norm_img[np.newaxis, :]
#                     norm_img_batch.append(norm_img)

#             norm_img_batch = np.concatenate(norm_img_batch)
#             norm_img_batch = norm_img_batch.copy()
#             norm_img_batch = norm_img_batch.astype(np.float32)  # FORCE float32


#             # if self.benchmark:
#             #     self.autolog.times.stamp()

#             if self.rec_algorithm == "SRN":
#                 encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
#                 gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
#                 gsrm_slf_attn_bias1_list = np.concatenate(gsrm_slf_attn_bias1_list)
#                 gsrm_slf_attn_bias2_list = np.concatenate(gsrm_slf_attn_bias2_list)
#                 inputs = [norm_img_batch, encoder_word_pos_list, gsrm_word_pos_list, gsrm_slf_attn_bias1_list, gsrm_slf_attn_bias2_list]
                
#                 if self.use_onnx:
#                     input_dict = {self.input_tensor.name: norm_img_batch}
#                     outputs = self.predictor.run(self.output_tensors, input_dict)
#                     preds = {"predict": outputs[2]}
#                 else:
#                     input_names = self.predictor.get_input_names()
#                     for i in range(len(input_names)):
#                         self.predictor.get_input_handle(input_names[i]).copy_from_cpu(inputs[i])
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = {"predict": outputs[2]}
#             elif self.rec_algorithm == "SAR":
#                 valid_ratios = np.concatenate(valid_ratios).astype(np.float32)
#                 inputs = [norm_img_batch, np.array([valid_ratios], dtype=np.float32).T]
#                 if self.use_onnx:
#                     input_dict = {self.input_tensor.name: norm_img_batch}
#                     outputs = self.predictor.run(self.output_tensors, input_dict)
#                     preds = outputs[0]
#                 else:
#                     input_names = self.predictor.get_input_names()
#                     for i in range(len(input_names)):
#                         self.predictor.get_input_handle(input_names[i]).copy_from_cpu(inputs[i])
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = outputs[0]
#             elif self.rec_algorithm == "RobustScanner":

#                 valid_ratios = np.concatenate(valid_ratios).astype(np.float32)
#                 word_positions_list = np.concatenate(word_positions_list).astype(np.int64)
#                 inputs = [norm_img_batch, valid_ratios, word_positions_list]
                
#                 if self.use_onnx:
#                     input_dict = {
#                         'x': norm_img_batch,
#                         'data_0': valid_ratios,
#                         'data_1': word_positions_list
#                     }
#                     print(input_dict, "inputs")
#                     outputs = self.predictor.run(None, input_dict)
#                     preds = outputs[0]

#                 else:
#                     input_names = self.predictor.get_input_names()
#                     for i in range(len(input_names)):
#                         self.predictor.get_input_handle(input_names[i]).copy_from_cpu(inputs[i])
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = outputs[0]
#             elif self.rec_algorithm == "CAN":
#                 norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
#                 word_label_list = np.concatenate(word_label_list)
#                 inputs = [norm_img_batch, norm_img_mask_batch, word_label_list]
#                 if self.use_onnx:
#                     input_dict = {self.input_tensor.name: norm_img_batch}
#                     outputs = self.predictor.run(self.output_tensors, input_dict)
#                     preds = outputs
#                 else:
#                     input_names = self.predictor.get_input_names()
#                     for i in range(len(input_names)):
#                         self.predictor.get_input_handle(input_names[i]).copy_from_cpu(inputs[i])
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = outputs
#             elif self.rec_algorithm == "LaTeXOCR":
#                 inputs = [norm_img_batch]
#                 if self.use_onnx:
#                     input_dict = {self.input_tensor.name: norm_img_batch}
#                     outputs = self.predictor.run(self.output_tensors, input_dict)
#                     preds = outputs
#                 else:
#                     input_names = self.predictor.get_input_names()
#                     for i in range(len(input_names)):
#                         self.predictor.get_input_handle(input_names[i]).copy_from_cpu(inputs[i])
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = outputs
#             else:
#                 if self.use_onnx:
#                     input_dict = {self.input_tensor.name: norm_img_batch}
#                     outputs = self.predictor.run(self.output_tensors, input_dict)
#                     preds = outputs[0]
#                 else:
#                     self.input_tensor.copy_from_cpu(norm_img_batch)
#                     self.predictor.run()
#                     outputs = [ot.copy_to_cpu() for ot in self.output_tensors]
#                     preds = outputs[0] if len(outputs) == 1 else outputs

#             if self.postprocess_params["name"] == "CTCLabelDecode":
#                 rec_result = self.postprocess_op(preds, return_word_box=self.return_word_box, wh_ratio_list=wh_ratio_list, max_wh_ratio=max_wh_ratio)
#             elif self.postprocess_params["name"] == "LaTeXOCRDecode":
#                 preds = [p.reshape([-1]) for p in preds]
#                 rec_result = self.postprocess_op(preds)
#             else:
#                 rec_result = self.postprocess_op(preds)

#             for rno in range(len(rec_result)):
#                 rec_res[indices[beg_img_no + rno]] = rec_result[rno]
#             # if self.benchmark:
#             #     self.autolog.times.end(stamp=True)

#         return rec_res, time.time() - st


# def _check_image_file(path):
#     img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
#     return any([path.lower().endswith(e) for e in img_end])


# def get_image_file_list(img_file, infer_list=None):
#     imgs_lists = []
#     if infer_list and not os.path.exists(infer_list):
#         raise Exception("not found infer list {}".format(infer_list))
#     if infer_list:
#         with open(infer_list, "r") as f:
#             lines = f.readlines()
#         for line in lines:
#             image_path = line.strip().split("\t")[0]
#             image_path = os.path.join(img_file, image_path)
#             imgs_lists.append(image_path)
#     else:
#         if img_file is None or not os.path.exists(img_file):
#             raise Exception("not found any img file in {}".format(img_file))

#         img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
#         if os.path.isfile(img_file) and _check_image_file(img_file):
#             imgs_lists.append(img_file)
#         elif os.path.isdir(img_file):
#             for single_file in os.listdir(img_file):
#                 file_path = os.path.join(img_file, single_file)
#                 if os.path.isfile(file_path) and _check_image_file(file_path):
#                     imgs_lists.append(file_path)

#     if len(imgs_lists) == 0:
#         raise Exception("not found any img file in {}".format(img_file))
#     imgs_lists = sorted(imgs_lists)
#     return imgs_lists

# def main(image_dir, rec_image_shape, rec_char_dict_path):
#     image_file_list = get_image_file_list(image_dir)
#     valid_image_file_list = []
#     img_list = []
#     text_recognizer = TextRecognizer(rec_image_shape=rec_image_shape, rec_char_dict_path=rec_char_dict_path)
#     for image_file in image_file_list:
#         img, flag, _ = check_and_read(image_file)
#         if not flag:
#             print(img, flag)
#             img = cv2.imread(image_file)
#         if img is None:
#             continue
#         valid_image_file_list.append(image_file)
#         img_list.append(img)
#     try:
#         rec_res, _ = text_recognizer(img_list)
#     except Exception as E:
#         print(E)
    

# if __name__ == "__main__":
#     main(image_dir = '/home/arshia/Downloads/plate_images/dataset_free/recognition/images/1_01_R_20250101100000_frame12145_jpg.rf.8e926bcb098c9066764f2624d6cb5a0c.jpg',
#          rec_image_shape='3,48,48,160',
#          rec_char_dict_path= './PaddleOCR/ppocr/utils/dict90.txt',)

import cv2
import numpy as np
import os
import re
import math
from PIL import Image
import time
import sys
import logging
import argparse
import onnxruntime as ort  # Import here to ensure availability

logger = logging.getLogger('ppocr')

# Define args as Namespace
args = argparse.Namespace(
    use_gpu=False,
    use_xpu=False,
    use_npu=False,
    use_mlu=False,
    use_gcu=False,
    ir_optim=True,
    use_tensorrt=False,
    min_subgraph_size=15,
    precision='fp32',
    gpu_mem=500,
    gpu_id=0,
    image_dir='/home/arshia/Downloads/plate_images/dataset_free/recognition/images/1_01_R_20250101100000_frame12145_jpg.rf.8e926bcb098c9066764f2624d6cb5a0c.jpg',
    page_num=0,
    det_algorithm='DB',
    det_model_dir=None,
    det_limit_side_len=960,
    det_limit_type='max',
    det_box_type='quad',
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    max_batch_size=10,
    use_dilation=False,
    det_db_score_mode='fast',
    det_east_score_thresh=0.8,
    det_east_cover_thresh=0.1,
    det_east_nms_thresh=0.2,
    det_sast_score_thresh=0.5,
    det_sast_nms_thresh=0.2,
    det_pse_thresh=0,
    det_pse_box_thresh=0.85,
    det_pse_min_area=16,
    det_pse_scale=1,
    scales=[8, 16, 32],
    alpha=1.0,
    beta=1.0,
    fourier_degree=5,
    rec_algorithm='RobustScanner',
    rec_model_dir='/home/arshia/Downloads/model.onnx',
    rec_image_inverse=True,
    rec_image_shape='3,48,48,160',
    rec_batch_num=6,
    max_text_length=25,
    rec_char_dict_path='ppocr/utils/dict90.txt',
    use_space_char=False,
    vis_font_path='./doc/fonts/simfang.ttf',
    drop_score=0.5,
    e2e_algorithm='PGNet',
    e2e_model_dir=None,
    e2e_limit_side_len=768,
    e2e_limit_type='max',
    e2e_pgnet_score_thresh=0.5,
    e2e_char_dict_path='./ppocr/utils/ic15_dict.txt',
    e2e_pgnet_valid_set='totaltext',
    e2e_pgnet_mode='fast',
    use_angle_cls=False,
    cls_model_dir=None,
    cls_image_shape='3, 48, 192',
    label_list=['0', '180'],
    cls_batch_num=6,
    cls_thresh=0.9,
    enable_mkldnn=False,
    cpu_threads=10,
    use_pdserving=False,
    warmup=False,
    sr_model_dir=None,
    sr_image_shape='3, 32, 128',
    sr_batch_num=1,
    draw_img_save_dir='./inference_results',
    save_crop_res=False,
    crop_res_save_dir='./output',
    use_mp=False,
    total_process_num=1,
    process_id=0,
    benchmark=False,
    save_log_path='./log_output/',
    show_log=True,
    use_onnx=True,
    onnx_providers=None,  # Changed from False to None as per typical usage
    onnx_sess_options=None,
    return_word_box=False
)

def create_predictor(args, mode, logger):
    if mode == "rec":
        model_dir = args.rec_model_dir
    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    if args.use_onnx:
        import onnxruntime as ort
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(model_file_path))
        sess_options = args.onnx_sess_options or None
        if args.onnx_providers and len(args.onnx_providers) > 0:
            sess = ort.InferenceSession(
                model_file_path,
                providers=args.onnx_providers,
                sess_options=sess_options,
            )
        elif args.use_gpu:
            sess = ort.InferenceSession(
                model_file_path,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"device_id": args.gpu_id, "cudnn_conv_algo_search": "DEFAULT"},
                    )
                ],
                sess_options=sess_options,
            )
        else:
            sess = ort.InferenceSession(
                model_file_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options,
            )
        inputs = sess.get_inputs()
        return (
            sess,
            inputs[0] if len(inputs) == 1 else [vo.name for vo in inputs],
            None,
            None,
        )

def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == "gif":
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        try:
            import fitz  # pyMuPDF
            from PIL import Image
            imgs = []
            with fitz.open(img_path) as pdf:
                for pg in range(0, pdf.page_count):
                    page = pdf[pg]
                    mat = fitz.Matrix(2, 2)
                    pm = page.get_pixmap(matrix=mat, alpha=False)
                    # if width or height > 2000 pixels, don't enlarge the image
                    if pm.width > 2000 or pm.height > 2000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    imgs.append(img)
                return imgs, False, True
        except ImportError:
            raise ImportError("Please install fitz: pip install pymupdf")
    return None, False, False

class BaseRecLabelDecode(object):
    """Convert between text-label and text-index"""
    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if "arabic" in character_dict_path:
                self.reverse = True
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
    def pred_reverse(self, pred):
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)
        return "".join(pred_re[::-1])
    def add_special_char(self, dict_character):
        return dict_character
    def get_word_info(self, text, selection):
        state = None
        word_content = []
        word_col_content = []
        word_list = []
        word_col_list = []
        state_list = []
        valid_col = np.where(selection == True)[0]
        for c_i, char in enumerate(text):
            if "\u4e00" <= char <= "\u9fff":
                c_state = "cn"
            elif bool(re.search("[a-zA-Z0-9]", char)):
                c_state = "en&num"
            else:
                c_state = "splitter"
            if (
                char == "."
                and state == "en&num"
                and c_i + 1 < len(text)
                and bool(re.search("[0-9]", text[c_i + 1]))
            ): # grouping floating number
                c_state = "en&num"
            if (
                char == "-" and state == "en&num"
            ): # grouping word with '-', such as 'state-of-the-art'
                c_state = "en&num"
            if state == None:
                state = c_state
            if state != c_state:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state
            if state != "splitter":
                word_content.append(char)
                word_col_content.append(valid_col[c_i])
        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)
        return word_list, word_col_list, state_list
    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
        return_word_box=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = "".join(char_list)
            if self.reverse: # for arabic rec
                text = self.pred_reverse(text)
            if return_word_box:
                word_list, word_col_list, state_list = self.get_word_info(
                    text, selection
                )
                result_list.append(
                    (
                        text,
                        np.mean(conf_list).tolist(),
                        [
                            len(text_index[batch_idx]),
                            word_list,
                            word_col_list,
                            state_list,
                        ],
                    )
                )
            else:
                result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
    def get_ignored_tokens(self):
        return [0] # for ctc blank

class SARLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""
    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(SARLabelDecode, self).__init__(character_dict_path, use_space_char)
        self.rm_symbol = kwargs.get("rm_symbol", False)
    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1
        return dict_character
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(self.end_idx):
                    if text_prob is None and idx == 0:
                        continue
                    else:
                        break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            if self.rm_symbol:
                comp = re.compile("[^A-Z^a-z^0-9^\u4e00-\u9fa5]")
                text = text.lower()
                text = comp.sub("", text)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list
    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, np.ndarray):  # No paddle.Tensor, use np
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
        else:
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label
    def get_ignored_tokens(self):
        return [self.padding_idx]

class TextRecognizer(object):
    def __init__(self,
                 rec_image_shape,
                 rec_char_dict_path,
                 ):
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self.rec_batch_num = 6
        self.rec_algorithm = 'RobustScanner'
        postprocess_params = {
            "name": "SARLabelDecode",
            "character_dict_path": rec_char_dict_path,
            "use_space_char": False,
            "rm_symbol": True,
        }
        self.postprocess_op = SARLabelDecode(**postprocess_params)
        self.return_word_box = False
        self.use_onnx = True
        # Call create_predictor here
        self.predictor, self.input_tensor, self.output_tensors, self.config = create_predictor(args, "rec", logger)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        width_divisor = int(1 / width_downsample_ratio)
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype("float32")
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape
        return padding_im, resize_shape, pad_shape, valid_ratio

    def __call__(self, img_list):
        img_num = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            valid_ratios = []
            word_positions_list = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "RobustScanner":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]],
                        self.rec_image_shape,
                        width_downsample_ratio=0.25,
                    )
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                    word_positions = np.array(range(0, 40)).astype("int64")
                    word_positions = np.expand_dims(word_positions, axis=0)
                    word_positions_list.append(word_positions)
                else:
                    norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            norm_img_batch = norm_img_batch.astype(np.float32)
            if self.rec_algorithm == "RobustScanner":
                valid_ratios = np.concatenate(valid_ratios).astype(np.float32)
                word_positions_list = np.concatenate(word_positions_list).astype(np.int64)
                inputs = [norm_img_batch, valid_ratios, word_positions_list]
                if self.use_onnx:
                    input_dict = {
                        'x': norm_img_batch,
                        'data_0': valid_ratios,
                        'data_1': word_positions_list
                    }
                    outputs = self.predictor.run(None, input_dict)
                    preds = outputs[0]
            else:
                if self.use_onnx:
                    input_dict = {'x': norm_img_batch}  # Assuming default input name 'x'
                    outputs = self.predictor.run(None, input_dict)
                    preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res, time.time() - st

def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])

def get_image_file_list(img_file, infer_list=None):
    imgs_lists = []
    if infer_list and not os.path.exists(infer_list):
        raise Exception("not found infer list {}".format(infer_list))
    if infer_list:
        with open(infer_list, "r") as f:
            lines = f.readlines()
        for line in lines:
            image_path = line.strip().split("\t")[0]
            image_path = os.path.join(img_file, image_path)
            imgs_lists.append(image_path)
    else:
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))
        img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
        if os.path.isfile(img_file) and _check_image_file(img_file):
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                file_path = os.path.join(img_file, single_file)
                if os.path.isfile(file_path) and _check_image_file(file_path):
                    imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def main(image_dir, rec_image_shape, rec_char_dict_path):
    image_file_list = get_image_file_list(image_dir)
    valid_image_file_list = []
    img_list = []
    text_recognizer = TextRecognizer(rec_image_shape=rec_image_shape, rec_char_dict_path=rec_char_dict_path)
    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)
        print(rec_res)  # Print results for debugging
    except Exception as E:
        print(E)

if __name__ == "__main__":
    main(image_dir = '/home/arshia/Downloads/plate_images/data_gray/test_data/night (1011)_plate1_aug1.jpg',
         rec_image_shape='3,48,48,160',
         rec_char_dict_path= './PaddleOCR/ppocr/utils/dict90.txt',)