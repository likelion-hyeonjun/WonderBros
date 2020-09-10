import torch 

class Test:

    def __init__(self, trained_model, test_loader, num_data):
        self.trained_model = trained_model
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_data = num_data

    def OverallAccuracy(self):
        self.trained_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.trained_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the %d test images: %d %%' % (self.num_data,
        100 * correct / total))

    def ClassAccuracy(self, classes):
        self.trained_model.eval()
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.trained_model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels)
                for i in range(len(c)): #batch size
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
