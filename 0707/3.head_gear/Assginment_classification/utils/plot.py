# ./utils/plot.py

import matplotlib.pyplot as plt

def plot_dataset_distribution(annotation):
    annotation['data set'].value_counts().plot(kind='bar')

    plt.title('Data set distribution')
    plt.xlabel('Data set')
    plt.ylabel('Count')

    for i in range(len(annotation['data set'].value_counts())):
        plt.text(i, annotation['data set'].value_counts()[i], annotation['data set'].value_counts()[i], ha='center', va='bottom')
    
    return plt.show()

def plot_image(image_paths, data, output, target):
    image = image_paths(image_paths['filepaths'][:3])
    plt.imshow(data)
    plt.text(2.0,0, f'output:{output}')
    plt.text(-10.0,0,f'target: {target}')
    plt.show()
    return plt.savefig('savefig_sample.png')