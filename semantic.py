#Practical task 1

import spacy
nlp = spacy.load('en_core_web_md')



print('''Following these steps:
● Create a file called semantic.py and run all the code extracts above.
● Write a note on what you noticed about the similarities between cat,
monkey and banana and think of an example of your own.
● Run the example file on with the simpler language model ‘en_core_web_sm’
and write a note on what you notice may be different from the model
'en_core_web_md''')
print('\n')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print('The similarity between car, monkey and banana respectively is as follows: ')
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print('\n')


tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
print('\n')


sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)


for sentence in sentences:
    
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence , "-" , similarity)
print('\n')


'''Write a note on what you noticed about the similarities between cat,
monkey and banana.'''

print('''There is a strong similarity of 0.59 between cat and monkey as both are animals.
There is also a similarity of 0.4 between monkey and banana as monkey like to eat banana''')
print('\n')


'''Think of an example of your own'''

word1=nlp('car')
word2=nlp('bmw')
word3=nlp('snake')

print('Similarity score between car, bmw and snake respectively is a follows: ')
print(word1.similarity(word2))
print(word1.similarity(word3))
print(word2.similarity(word3))
print('\n')

print('''There is a strong similarity between car and bmw as bmw manufactures cars.
There is not much similarity between bmw and snake, car and snake as one is non living and 
the other is living organism respectively''')
print('\n')



'''Run the example file on with the simpler language model en_core_web_sm
and write a note on what you notice may be different from the model en_core_web_md'''

# Please read all the comments in this example file and all others.

import spacy  # importing spacy
nlp = spacy.load('en_core_web_sm') # using a smaller model instead of medium. 

# Now we are going to look into longer texts and compare them. 
# Below we have two lists: one containing complaints submitted to a company, and another of recipes found online.
# We want to establish how spaCy's model can identify similarities or dissimilarities between complaint and recipes. 

# Make sure to run this example file and read through the explanations.

# Below is a list of six complaints.
complaints = [ 'We bought a house in  CA. Our mortgage was handled by a company called ki. Soon after the mortgage was sold to ABC. Shortly after that XYZ took over the mortgage. The other day we got a notice not to send our payment to them but to loi instead. This is all so frustrating and wreaks of the  mortgage nightmare.',
'I got approved for a loan to buy a house I have submitted everything I need to for them I paid for the inspection and paid good faith check after all of that they said I did not get approved for the loan to cancel my contract because they do not want to wait for the down payments assistant said that the Sellers do not want to wait that long I feel like they are getting over on me I feel that they should have told me that I did not get approved before I spent my money and picked out a house Carrington mortgage in Ohio ',
'As per the correspondence, I received from : The University  This is to inform you that I have recently pulled my credit report and noticed that there is a collection listing from The University  on my credit report. I WAS never notified of this collection action or that I owed the debt. This letter is to inform you that I would like a verification of the debt and juilo ability to collect this money from me.',
'I am writing to dispute the follow information in my file.ON BOTH TransUnion & . for {$15000.00}. I have contacted this agency to advise to STOP CALLING ME this case was dismissed in court  2014. Please see the attached document from  County State Court. Thanking you in advanced regarding this matter.',
'I have not had a XXXX phone since early 2007. I have tried to resolve my bill in the past but it keeps reposting an old bill. I have no way to provide financial info from 8 years ago and they know that so they want me to prove it to them but I have no way to do that. Is there anyway to get  to find out how old it is.',
'I posted dated a check and mailed it for 2015 for my mortgage payment as my mortgage company will only take online payments if all the late charges are paid at once ( also illegal ), and the check was cashed on 2015 which cost me over {$70.00} in over draft fees with my bank.'
]

# We will now compare the similarity of the complaints to ascertain if spaCy's similarity
# model is able to distinguish between these long pieces of text.

print("-------------Complaints similarity---------------")
for token in complaints:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))
print('\n')

# Below is a list of six recipe instructions.

recipes= [ 'Bake in the preheated oven, stirring every 20 minutes, until sugar mixture has baked and caramelized onto popcorn and cashews, about 1 hour. Spread cashew caramel corn onto a parchment paper-lined baking sheet to cool. If desired, form into balls while still warm.',
'Combine brown sugar, corn syrup, butter, salt, and cream of tartar in a large saucepan. Bring to a boil, stirring constantly, until a candy thermometer inserted into the middle of the syrup, not touching the bottom, reads 260 degrees F (127 degrees C), 6 to 8 minutes.',
'Lift marshmallow fudge out of the pan by the edges of the foil and place on a large cutting board. Dip a large knife in the remaining confectioners\' sugar and slice fudge into 1 1/2-inch squares, continually dipping knife in the sugar after each slice.',
'Melt butter in a medium saucepan over medium heat; stir in condensed milk. Pour in chocolate chips; cook and stir until melted, 5 to 10 minutes.',
'Lightly grease a cookie sheet. Deflate the dough and turn it out onto a lightly floured surface. Roll the marzipan into a rope and place it in the center of the dough. Fold the dough over to cover it; pinch the seams together to seal. Place the loaf, seam side down, on the prepared baking sheet. Cover with a damp cloth and let rise until doubled in volume, about 40 minutes. Meanwhile, preheat oven to 350 degrees F (175 degrees C)',
'In a large bowl, cream together the butter, brown sugar, and white sugar. Beat in the instant pudding mix until blended. Stir in the eggs and vanilla. Blend in the flour mixture. Finally, stir in the chocolate chips and nuts. Drop cookies by rounded spoonfuls onto ungreased cookie sheets.'
]

# We will now compare the similarity of the recipes. to ascertain how well spaCy's similarity
# model is able to distinguish between them.

print("-------------Recipes similarity---------------")
for token in recipes:
    token = nlp(token)
    for token_ in recipes:
        token_ = nlp(token_)
        print(token.similarity(token_))

# Now we want to obtain the extent of similarity between the complaints and the recipes.
# we will loop through every recipe instruction and compare it with a complaint.

print("-------------Recipes similarity---------------")
print('\n')


for token in recipes:
    token = nlp(token)
    for token_ in complaints:
        token_ = nlp(token_)
        print(token.similarity(token_))
print('\n')
# What do you observe?
print('After using ‘en_core_web_sm’, the similarity score has reduced')
print('\n')
        


#Practical task 2
        
# Open the file in read mode
with open(r"C:\Users\Saad Amin\Desktop\Data Science material\CoGrammer-Bootcamp\Data Science (Fundamentals)\Data Science (Fundamentals)\T20 - NLP - Semantic Similarity\14-002 NLP - Semantic Similarity\movies.txt", 'r') as file:
    # Read the file contents
    movie_file = file.read()
    # Print the contents
    print(movie_file)
print('\n')


new_movie_file=movie_file.split('\n')
print(new_movie_file)
print('\n')


#making a dictionary from txt file, with movie names as 'keys' and their description as 'values'

import re

dict_movie = {}
for movie in new_movie_file:
  
  match = re.search(r"(.*):", movie)
    
  if match:
    movie_name = match.group(1).strip()  # Removing whitespaces from movie name
    description = movie[len(movie_name)+2:]  # Description is extracted after colon and spaces
    dict_movie[movie_name] = description
  else:
    print("Movie name not found for a line:", movie)

print(dict_movie)
print('\n')



'''Creating a function recommend_movie to return which movies a user would watch by 
getting their description of movie'''

nlp = spacy.load('en_core_web_md')   #loading medium package

def recommend_movie(description):
    
    nlp_description= nlp(description)   #processing the data
    nlp_description
    

    new_movie_dict={}
    
    for movies_names, movies_description in dict_movie.items():
        
        similarity_score= nlp(movies_description).similarity(nlp_description)  #getting similarity_score between description of movies and the description that I provided
        new_movie_dict[movies_names]=similarity_score    #making a dictionary with key as movie name and similarity_score as values
        
    #print(new_movie_dict)

    return max(new_movie_dict, key=new_movie_dict.get)   #returning the movie with maximum similarity_score


descrip=input('Write description of the movie you would like to watch: ')
print('Recommended movies is : ', recommend_movie(description=descrip))