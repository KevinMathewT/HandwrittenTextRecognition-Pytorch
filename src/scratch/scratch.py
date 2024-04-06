import warnings

import cv2
import matplotlib.pyplot as plt

# from .segment import segment_lines
from .utils import *
from .validate_segment import create_df

warnings.filterwarnings("ignore")


def display_lines(lines_arr, orient='vertical'):
    plt.figure(figsize=(30, 30))
    if orient not in ['vertical', 'horizontal']:
        raise ValueError(
            "Orientation is on of 'vertical', 'horizontal', defaul = 'vertical'")
    if orient == 'vertical':
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)  # A grid of 2 rows x 10 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            # to hide tick values on X and Y axis
            plt.xticks([]), plt.yticks([])
    else:
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(40, 1, i+1)  # A grid of 40 rows x 1 columns
            plt.axis('off')
            plt.title("Line #{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            # to hide tick values on X and Y axis
            plt.xticks([]), plt.yticks([])
    plt.show()


def segment_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", img)
    cv2.waitKey()
    form = cv2.GaussianBlur(img, ksize=(49, 49), sigmaX=11, sigmaY=11)
    ret, form = cv2.threshold(form, 215, 255, cv2.THRESH_BINARY)
    form = 255 - form

    _, labels_im = cv2.connectedComponents(form)

    min_x = {}
    max_x = {}
    min_y = {}
    max_y = {}

    for i in range(labels_im.shape[0]):
        for j in range(labels_im.shape[1]):
            if not labels_im[i][j] == 0:
                if labels_im[i][j] not in min_x.keys():
                    min_x[labels_im[i][j]] = i
                    max_x[labels_im[i][j]] = i
                    min_y[labels_im[i][j]] = j
                    max_y[labels_im[i][j]] = j
                else:
                    min_x[labels_im[i][j]] = min(min_x[labels_im[i][j]], i)
                    max_x[labels_im[i][j]] = max(max_x[labels_im[i][j]], i)
                    min_y[labels_im[i][j]] = min(min_y[labels_im[i][j]], j)
                    max_y[labels_im[i][j]] = max(max_y[labels_im[i][j]], j)

    bounding_boxes = []
    for i in range(1, len(min_x.keys()) + 1):
        bounding_boxes.append((min_x[i], max_x[i], min_y[i], max_y[i]))

    bounding_boxes = sorted(bounding_boxes, key=lambda entry: entry[0])
    lines = []

    current_window_min = -1
    current_window_max = -1
    current_window_str = -1

    for i in range(len(bounding_boxes)):
        if current_window_str == -1:
            current_window_min = bounding_boxes[i][0]
            current_window_max = bounding_boxes[i][1]
            current_window_str = i
            continue
        if (i == len(bounding_boxes) - 1) or (not (
                (bounding_boxes[i][0] >= current_window_min and bounding_boxes[i][0] <= current_window_max) or (
                bounding_boxes[i][1] >= current_window_min and bounding_boxes[i][1] <= current_window_max))):
            if i == len(bounding_boxes) - 1:
                i = i + 1
            bounding_boxes[current_window_str:i] = sorted(bounding_boxes[current_window_str:i],
                                                          key=lambda entry: entry[2])

            curr_win_min_x, curr_win_max_x, curr_win_min_y, curr_win_max_y = bounding_boxes[current_window_str]
            for j in range(current_window_str, i):
                curr_win_min_x = min(curr_win_min_x, bounding_boxes[j][0])
                curr_win_max_x = max(curr_win_max_x, bounding_boxes[j][0])
                curr_win_min_y = min(curr_win_min_y, bounding_boxes[j][0])
                curr_win_max_y = max(curr_win_max_y, bounding_boxes[j][0])
            lines.append([curr_win_min_x, curr_win_max_x, curr_win_min_y, curr_win_max_y])

            if i == len(bounding_boxes):
                i = i - 1
            current_window_min = bounding_boxes[i][0]
            current_window_max = bounding_boxes[i][1]
            current_window_str = i

        if (bounding_boxes[i][0] >= current_window_min and bounding_boxes[i][0] <= current_window_max) or (
                bounding_boxes[i][1] >= current_window_min and bounding_boxes[i][1] <= current_window_max):
            current_window_min = min(current_window_min, bounding_boxes[i][0])
            current_window_max = min(current_window_max, bounding_boxes[i][1])

    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    print(lines)

    final_lines = []
    for i in range(len(lines)):
        final_lines.append(img[lines[i][0]:lines[i][1], lines[i][2]:lines[i][3]])

    segments = []
    for i in range(len(bounding_boxes)):
        segments.append(img[bounding_boxes[i][0]:bounding_boxes[i][1], bounding_boxes[i][2]:bounding_boxes[i][3]])

    return final_lines


# if os.path.isfile(config.FORMS_DF) and False:
#     print(f"Loaded cached FORMS_DF from {config.FORMS_DF}")
#     df = pd.read_csv(config.FORMS_DF)
# else:
#     df = create_df()
#     df.to_csv(config.FORMS_DF, index=False)

# dataset = HandWritingFormsDataset(df, transforms=get_valid_transforms())[:1]
# dataloader = DataLoader(
#         dataset,
#         batch_size=config.VALID_SEGMENT_BATCH_SIZE,
#         drop_last=config.DROP_LAST,
#         num_workers=config.CPU_WORKERS,
#         shuffle=False)

# lines = []
# for line in dataset[0][0]:
#     lines.append(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY))
#     print(cv2.cvtColor(line.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY).shape)

image_id = "a01-000x"
df = create_df()
print(df)
row = df[df["image_id"] == image_id].reset_index(drop=True)
print(row)
bb = [row['x1'][0], row['y1'][0],
      row['x2'][0], row['y2'][0]]
print(bb)
print(row)
img = get_img("D:\\Kevin\\Machine Learning\\IAM Dataset Full\\original\\forms\\" + image_id + ".png")
img = img[bb[1]:bb[3], bb[0]:bb[2], :]

#######################

# image = img
# # cv2.imshow('orig', image)
# # cv2.waitKey(0)
#
# # grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # cv2.imshow('gray', image)
# # cv2.waitKey(0)
#
# # binary
# ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
# # cv2.imshow('second', image)
# # cv2.waitKey(0)
#
# # dilation
# kernel = np.ones((5, 15), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# # cv2.imshow('dilated', image)
# # cv2.waitKey(0)
#
# # find contours
# ctrs, hier = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # sort contours
# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
#
#
# for i, ctr in enumerate(sorted_ctrs):
#     # Get bounding box
#     x, y, w, h = cv2.boundingRect(ctr)
#
#     # Getting ROI
#     roi = image[y:y+h, x:x+w]
#
#     # show ROI
#     # cv2.imshow('segment no:'+str(i),roi)
#     cv2.imwrite("segment_no_"+str(i)+".png",roi)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
#     # cv2.waitKey(0)
#
# # cv2.imwrite('final_bounded_box_image.png',image)
# cv2.imshow('marked areas', image)
# cv2.waitKey(0)

#######################

lines = segment_image(img)
print(len(lines))
#
for i in range(len(lines)):
    cv2.imshow(f'segment {i}', lines[i])
    cv2.waitKey(0)

# display_lines(lines)

# print(len(dataset))
# print(len(dataset[0]))
# print(dataset[0][1])
# print(len(dataset[0][0]))

# for i in range(len(dataset[0][0])):
#     plt.imshow(np.transpose(
#         dataset[0][0][i].cpu().detach().numpy(), (1, 2, 0)))
#     plt.show()
# img = get_img("D:\Kevin\Machine Learning\IAM Dataset Full\original\\forms\\a01-000u.png")
# print(img.shape)
# plt.imshow(img)
# print(img)
# plt.show()

# net = get_net(name=config.NET)
# img = dataset[0][0]
# print(img.size())
# print(net(img).size())
#
# test_pipeline(net, None, dataloader, get_device(n=0))

