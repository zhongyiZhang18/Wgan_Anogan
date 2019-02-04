import Wgan_anogan
anogan = Wgan_anogan
### 2. test generator
generated_img = anogan.generate(25)
img = anogan.combine_images(generated_img)
img = (img*127.5)+127.5
img = img.astype(np.uint8)
img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

### opencv view
#cv2.namedWindow('generated', 0)
#cv2.resizeWindow('generated', 256, 256)
#cv2.imshow('generated', img)
cv2.imwrite('Weights/generator.png', img)
#cv2.waitKey()

### plt view
plt.figure(num=0, figsize=(4, 4))
plt.title('trained generator')
plt.imshow(img, cmap=plt.cm.gray)
plt.show()

#exit()

### 3. other class anomaly detection


def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 84, 60, 1), iterations=500, d=d)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(84,60,1) - similar_img.reshape(84,60,1)
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(84,60,1)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(84,60,1)*127.5+127.5).astype(np.uint8)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.5, residual_color, 0.5, 0.)
    cv2.imwrite('./unseenInput_color.png', original_x_color)
    cv2.imwrite('./np_residual.png', np_residual)

    cv2.imwrite('./residual_color.png', residual_color)


    return ano_score, original_x, similar_x, show



img_idx = 3
label_idx = 3
start = cv2.getTickCount()
score, qurey, pred, diff = anomaly_detection(test_img1)
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
cv2.imwrite('./qurey1.png', qurey)
cv2.imwrite('./pred1.png', pred)
cv2.imwrite('./diff1.png', diff)

## matplot view
plt.figure(1, figsize=(2, 2))
plt.title('unseen1 image')
plt.imshow(qurey.reshape(84,60), cmap=plt.cm.gray)

print("anomaly score : ", score)
plt.figure(2, figsize=(2, 2))
plt.title('generated similar1 image')
plt.imshow(pred.reshape(84,60), cmap=plt.cm.gray)

plt.figure(3, figsize=(2, 2))
plt.title('anomaly detection1')
plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
plt.show()





start = cv2.getTickCount()
score, qurey, pred, diff = anomaly_detection(test_img2)
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d label, %d : done'%(label_idx, img_idx), '%.2f'%score, '%.2fms'%time)
cv2.imwrite('./qurey2.png', qurey)
cv2.imwrite('./pred2.png', pred)
cv2.imwrite('./diff2.png', diff)

## matplot view
plt.figure(1, figsize=(2, 2))
plt.title('unseen2 image')
plt.imshow(qurey.reshape(84,60), cmap=plt.cm.gray)

print("anomaly score : ", score)
plt.figure(2, figsize=(2, 2))
plt.title('generated similar2 image')
plt.imshow(pred.reshape(84,60), cmap=plt.cm.gray)

plt.figure(3, figsize=(2, 2))
plt.title('anomaly detection2')
plt.imshow(cv2.cvtColor(diff,cv2.COLOR_BGR2RGB))
plt.show()