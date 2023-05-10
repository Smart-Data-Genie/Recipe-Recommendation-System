
import discord
import os
from discord.ext import commands
import asyncio
import transformers
from transformers import BertTokenizer, BertForMaskedLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import spacy

client =  discord.Client(intents=discord.Intents.default())
bot = commands.Bot(command_prefix='.', intents=discord.Intents.default())

test_recipes= '/content/drive/MyDrive/Colab Notebooks/NLP/layer1.json'

with open(test_recipes, 'r') as f:
    test_recipes = json.load(f)
from transformers import BertTokenizer, BertForMaskedLM
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BERT model for masked LM
model = BertForMaskedLM.from_pretrained('bert-base-uncased')


# Load the saved model weights
model= model.from_pretrained('/content/drive/MyDrive/Colab Notebooks/NLP/epoch1_batch9000')

model= model.to(device)

# Load fine-tuned BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('/content/drive/MyDrive/Colab Notebooks/NLP/tokenizer')

important_nutrients = [
              "Protein",
              "Total lipid (fat)",
              "Carbohydrate, by difference",
              "Fiber, total dietary",
              "Sugars, total including NLEA",
              "Calcium, Ca",
              "Iron, Fe",
              "Magnesium, Mg",
              "Phosphorus, P",
              "Potassium, K",
              "Sodium, Na",
              "Zinc, Zn",
              "Vitamin C, total ascorbic acid"
          ]


def get_recipe (user_input):
  encoded_user_input = tokenizer.encode_plus(user_input, 
                                  add_special_tokens=True, 
                                  max_length=512, 
                                  padding='max_length', 
                                  truncation= True,
                                  return_attention_mask=True, 
                                  return_tensors='pt').to(device)

# Get the BERT embeddings for the user input
  with torch.no_grad():
      try:
          user_output = model(**encoded_user_input)
          user_embedding = torch.mean(user_output[0], dim=1)
      except Exception as e:
          print(f"Error getting user embedding: {e}")
          user_embedding = None
                      
  # Get the BERT embeddings for the recipe text
  recipe_embeddings = []
  for recipe in test_recipes[:500]:
      title = recipe['title']
      ingredients = [ingr['text'] for ingr in recipe['ingredients']]
      instructions = [instr['text'] for instr in recipe['instructions']]
      recipe_text = title + ' ' + ' '.join(ingredients) + ' ' + ' '.join(instructions)

      encoded_inputs = tokenizer.encode_plus(recipe_text, 
                                              add_special_tokens=True, 
                                              max_length=512, 
                                              padding='max_length', 
                                              truncation= True,
                                              return_attention_mask=True, 
                                              return_tensors='pt').to(device)
      with torch.no_grad():
          try:
              output = model(**encoded_inputs)
              recipe_embedding = torch.mean(output[0], dim=1)
              recipe_embeddings.append(recipe_embedding)
          except Exception as e:
              print(f"Error getting embedding for recipe {title}: {e}")    

  if user_embedding is not None:
          # Compute the cosine similarity between the user input and each recipe
          similarities = cosine_similarity(user_embedding.cpu(), torch.cat(recipe_embeddings).cpu(), dense_output=True)

          # Get the index of the top most similar recipe
          top_index = similarities.argsort()[0][::-1][0]

          # Concatenate the title, ingredients and instructions into a single message
          recipe = test_recipes[top_index]
          title = recipe['title']
          ingredients = [ingr['text'] for ingr in recipe['ingredients']]
          instructions = [instr['text'] for instr in recipe['instructions']]
          nlp = spacy.load("en_core_web_sm")
          ingredient_names = []
          # dictionary to store total nutrient values
          total_nutrients = {}

          for ingredient in ingredients:
              doc = nlp(ingredient)
              name = ""
              for token in doc:
                  if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                      if name != "":
                          name += " "
                      name += token.text
              if "(" in name:
                  name = name[:name.index("(")].strip()
              if "," in name:
                  name = name[:name.index(",")].strip()
              if "to " in name:
                  name = name[:name.index("to ")].strip()
              if name.endswith("s"):
                  name = name[:-1]
              if "recipe" in name:
                  name = name[:name.index("recipe")].strip()
              if "page" in name:
                  name = name[:name.index("page")].strip()
              if "Note" in name:
                  name = name[:name.index("Note")].strip()
              ingredient_names.append(name)
          # print(ingredient_names)

          # make request to USDA FoodData Central API for each ingredient and add the nutrient values to the total_nutrients dictionary
          for ingredient in ingredient_names:
              response = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params={"api_key": "wrOUlYRyTJUOBsGxpYlQPfOwIeukbnsVec7rrvGv", "query": ingredient})
              nutrients = response.json()["foods"][0]["foodNutrients"]
              for nutrient in nutrients:
                  nutrient_name = nutrient['nutrientName']
                  nutrient_number = float(nutrient['nutrientNumber'])
                  unit_name = nutrient['unitName']
                  if nutrient_name in total_nutrients:
                      total_nutrients[nutrient_name] += nutrient_number
                  else:
                      total_nutrients[nutrient_name] = nutrient_number

          # print the total nutrient values
          nutrient_message = ''
          print("\nNutrients in this Food Recipe are:\n")
          for nutrient_name, nutrient_number in sorted(total_nutrients.items()):
              unit_name = nutrients[0]['unitName']
              if nutrient_name in important_nutrients:
                  nutrient_message += f"{nutrient_name}, "

          # create a recipe message containing title, ingredients, instructions, and nutrient information
          #recipe_message = f"Title: {title}\n\nIngredients:\n" + '\n'.join(ingredients) + "\n\nInstructions:\n" + '\n'.join(instructions) 
          nutrient_message = f"\n\nThe nutrients in this food recipe are:\n" + nutrient_message[:-2]
          return  title, ingredients, instructions, nutrient_message

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.lower() in ('hello', 'hi', 'hey', '', 'what''s up'):
        await message.channel.send('Hello! I am your Recipe Recommendation Bot. What do you want to eat today?')
        # Wait for user response
        try:
            response_message = await client.wait_for('message', check=lambda m: m.author == message.author and m.channel == message.channel, timeout=60.0)
            user_input = response_message.content 
            await message.channel.send("Finding a perfect recipe for you!..... ")   
            title, ingredients, instructions, nutrient_message = get_recipe(user_input)
            await message.channel.send("Here's a recipe you might like")    
            await message.channel.send("Title")
            await message.channel.send(title)
            await message.channel.send("Ingredients")
            await message.channel.send(ingredients)
            await message.channel.send("Instructions")
            await message.channel.send(instructions)
            await message.channel.send("The nutrition you are getting from this receipe are:")
            await message.channel.send(nutrient_message)

        except asyncio.TimeoutError:
            await message.channel.send("Sorry, you took too long to respond")
            return

        while True:
            # Ask if the user wants to browse more recipes
            await message.channel.send("Would you like to browse more recipes? (yes or no)")
            try:
                response_message = await client.wait_for('message', check=lambda m: m.author == message.author and m.channel == message.channel, timeout=60.0)
                user_response = response_message.content.lower()
                if user_response == 'yes':
                    await message.channel.send("What else would you like to eat?")
                    # Wait for user response
                    try:
                        response_message = await client.wait_for('message', check=lambda m: m.author == message.author and m.channel == message.channel, timeout=60.0)
                        user_input = response_message.content 
                        await message.channel.send("Finding a perfect recipe for you!..... ")   
                        title, ingredients, instructions, nutrient_message = get_recipe(user_input)
                        await message.channel.send("Here's a recipe you might like")    
                        await message.channel.send("Title")
                        await message.channel.send(title)
                        await message.channel.send("Ingredients")
                        await message.channel.send(ingredients)
                        await message.channel.send("Instructions")
                        await message.channel.send(instructions)
                        await message.channel.send("The nutrition you are getting from rhis receipe are:")
                        await message.channel.send(nutrient_message)
                    except asyncio.TimeoutError:
                        await message.channel.send("Sorry, you took too long to respond")
                        return
                elif user_response == 'no':
                    await message.channel.send("Alright, have a great day!")
                    return
                else:
                    await message.channel.send("Sorry, I didn't understand that. Please respond with 'yes' or 'no'.")
            except asyncio.TimeoutError:
                await message.channel.send("Sorry, you took too long to respond. Have a great day!")
                return      
client.run('MTEwMjMxOTMwMDU2NTU0MTA0NA.GEgZGg.qGKFCjZo5w_09H_qO54YkYye61XBrHD6ShKV98')