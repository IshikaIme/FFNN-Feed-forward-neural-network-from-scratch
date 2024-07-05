import pickle
model = pickle.load(open('model_FNN.pickle', 'rb'))
#predicted_labels = model.prediction_of_whole_dataset(x_test)
#accuracy = model.get_accuracy(predicted_labels, y_test)
#print(f"Accuracy on test set: {accuracy:.3f}")