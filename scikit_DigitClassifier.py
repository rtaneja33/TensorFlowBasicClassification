import matplotlib.pyplot as plt 
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()
#print(digits)

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
## above, we are plotting first 4 images in digits.images next to their labels from digits.target
plt.show()

## now lets apply a classifier 
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
## flattens data, turns into a (samples, feature) matrix

classifier = svm.SVC(gamma=0.001)
halfofdataset = n_samples // 2
classifier.fit(data[:halfofdataset], digits.target[:halfofdataset])
## lets run a classifier to fit half of our data, predict second half

expected = digits.target[halfofdataset:]
predicted = classifier.predict(data[halfofdataset:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
## Confusion matrix ?? Explain

images_and_predictions = list(zip(digits.images[halfofdataset:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
