from tensorflow import keras
import re
import numpy as np

def main():
    #retrieve user input and convert to all lowercase letters
    print('input text for prediction')
    text = input('----> ')
    text = text.lower() 
    
    def to_numbs(word):
        #iterate through each character in the word and assign it a numerical value. value will be added to list
        list = []
        for char in enumerate(word):
            if char[1] == 'a': list.append(1)
            if char[1] == 'b': list.append(2)
            if char[1] == 'c': list.append(3)
            if char[1] == 'd': list.append(4)
            if char[1] == 'e': list.append(5)
            if char[1] == 'f': list.append(6)
            if char[1] == 'g': list.append(7)
            if char[1] == 'h': list.append(8)
            if char[1] == 'i': list.append(9)
            if char[1] == 'j': list.append(10)
            if char[1] == 'k': list.append(11)
            if char[1] == 'l': list.append(12)
            if char[1] == 'm': list.append(13)
            if char[1] == 'n': list.append(14)
            if char[1] == 'o': list.append(15)
            if char[1] == 'p': list.append(16)
            if char[1] == 'q': list.append(17)
            if char[1] == 'r': list.append(18)
            if char[1] == 's': list.append(19)
            if char[1] == 't': list.append(20)
            if char[1] == 'u': list.append(21)
            if char[1] == 'v': list.append(22)
            if char[1] == 'w': list.append(23)
            if char[1] == 'x': list.append(24)
            if char[1] == 'y': list.append(25)
            if char[1] == 'z': list.append(26)
            if char[1] == 'á': list.append(27)
            if char[1] == 'é': list.append(28)
            if char[1] == 'ó': list.append(29)
            if char[1] == 'ú': list.append(30)
            if char[1] == 'í': list.append(31)
            if char[1] == 'ñ': list.append(32)

        #if the length of the list (word) is less than 10 characters, add Null values so that model can interpre it
        for i in range(0,9):
            if len(list) < 10: 
                list.append(-1) #because of the ReLu activation function, all negative numbers will be considered 0
            if len(list) == 10:
                return list
        return list

    re.sub('[^A-Za-z0-9]+', '', text) #remove all special characters from inputted text
    words_list = text.split() #converts text to a list of the individual words it contains

    #for each word in the list, convert it to numbers so that the model can process it correctly 
    list_of_lists = []
    for word in words_list:
        var = to_numbs(word) #converts to list ... var = list of numbers
        list_of_lists.append(var)

    #convert the list of lists to a numpy array -->> keras models only able to process arrays
    model_input = np.array(list_of_lists)        

    #load the pre-trained model and use it to make a prediction based on the array
    model = keras.models.load_model('ann_updated')
    predictions = model.predict(model_input)
    #model.predict returns a number between 0 and 1, 0 being Spanish and 1 being English

    #find the sum of each prediction to find the sentence average
    total = 0
    for i in enumerate(predictions):
        total = total + i[1]
    average = total/len(predictions)

    #round the average, and print the model's confidence in its selection 
    if average[0] < 0.50000000:
        print('Spanish: ', (1-average[0])*100, '%')
    else:
        print('English: ', (average[0])*100, '%' )


main() #I decided to functionize it to improve scalability