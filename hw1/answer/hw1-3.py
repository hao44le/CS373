import csv
import matplotlib as mpl
mpl.use('Agg')
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size
import matplotlib.pyplot as plt

def read_csv_and_fill_dict(csv_name,multiple_value=False):
  csvfile = open(csv_name,'rb')
  spamreader = csv.reader(csvfile)
  result_dict = dict()
  for row in spamreader:
    first,second = row
    if first in result_dict:
      result_dict[first].append(second)
    else:
      if multiple_value:
        result_dict[first] = [second]
      else:
        result_dict[first] = second
  csvfile.close()
  return result_dict

#Q3 a)
marvel_txt = "Marvel-data/marvel.csv"
character_comic_dict = read_csv_and_fill_dict(marvel_txt,True)

total_number_of_unique_characters = len(character_comic_dict)
highest_character_appearance = 0


for character_id in character_comic_dict:
  #print("{0}: {1}".format(character_id,len(character_comic_dict[character_id])))
  appearance = len(character_comic_dict[character_id])
  highest_character_appearance = max(highest_character_appearance,appearance)

x = []
y = []
for index in range(0,highest_character_appearance+1):
  x.append(index)
  y_value = 0.0
  for character_id in character_comic_dict:
    appearance = len(character_comic_dict[character_id])
    if appearance > index:
      y_value += 1
  y.append(y_value)
y = [y_value / total_number_of_unique_characters for y_value in y]

plt.xlim([1,max(x)])
plt.xlabel("comic characters appearance", fontsize=18)
plt.title("(ECCDF) of comic characters appearances in comic books (log-log)")
plt.ylabel("ECCDF", fontsize=18)
plt.loglog(x,y,"ro")
plt.savefig('hw1_3.comic_ECCDF_plot.png')

marvel_characters_txt = "Marvel-data/marvelCharacters.csv"
characters_name_dict = read_csv_and_fill_dict(marvel_characters_txt)

#3.b) (i)
quill_character_id = 0
for character_id in characters_name_dict:
  #print("{0}: {1}".format(character_id,characters_name_dict[character_id]))
  #break
  if characters_name_dict[character_id] == 'QUILL':
    quill_character_id = character_id
    break

print("\tquill_character_id:{0}".format(quill_character_id))
quill_appearance_array = []
for character_id in character_comic_dict:
  if character_id == quill_character_id:
    quill_appearance_array = character_comic_dict[character_id]
print("\tb.(i) QUILL appear in {0} distinct comic books".format(len(quill_appearance_array)))


#3.b (ii)
comic_character_dict = dict()
most_characters = 0
most_character_comic_id = 0
for character_id in character_comic_dict:
  for comic in character_comic_dict[character_id]:
    if comic not in comic_character_dict:
      comic_character_dict[comic] = [character_id]
    else:
      comic_character_dict[comic].append(character_id)
    if len(comic_character_dict[comic]) > most_characters:
      most_characters = len(comic_character_dict[comic])
      most_character_comic_id = comic

marvel_comicBooks_txt = "Marvel-data/marvelComicBooks.csv"
comic_name_dict = read_csv_and_fill_dict(marvel_comicBooks_txt)
most_character_comic_book_name = comic_name_dict[most_character_comic_id]
print("\tb.(ii) comic book that has the most number of characters is {0}. Its id is {1} with {2} # of characters".format(most_character_comic_book_name,most_character_comic_id,most_characters))
