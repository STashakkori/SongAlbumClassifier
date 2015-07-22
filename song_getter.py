__author__ = 'Sina Tashakkori'

from  pyechonest import song
from pyechonest import config
import json

"""
    main :: a method that connects to echonest and requests songs using an artist_map and a song_map.
"""
def main():
    config.ECHO_NEST_API_KEY="SCD9MTKKWSTGWCPCE"

    album_artist_map = {"Parachutes" : "coldplay",
                  "A Rush of Blood to the Head" : "coldplay",
                  "X & Y" : "coldplay",
                  "Viva la Vida or Death and All His Friends" : "coldplay",
                  "Mylo Xyloto" : "coldplay",
                  "Highway To Hell" : "ac/dc",
                  "The Foundation" : "zac brown band",
                  "If You're Reading This It's Too Late" : "drake",
                  "Random Access Memories" : "daft punk"
                  }


    album_song_map = {"Parachutes" : ["don't panic", "shiver", "spies", "sparks", "yellow", "trouble", "parachutes",
                                "high speed", "we never change", "everything's not lost"],
                "A Rush of Blood to the Head" : ["politik", "in my place", "god put a smile upon your face",
                                                 "the scientist", "clocks", "daylight", "green eyes", "warning sign",
                                                 "a whisper", "a rush of blood to the head", "amsterdam"],
                "X & Y" : ["square one", "what if", "white shadows", "fix you", "talk", "x & y", "speed of sound",
                         "a message", "low", "the hardest part", "swallowed in the sea", "twisted logic",
                         "til kingdom come"],
                "Viva la Vida or Death and All His Friends" : ["life in technicolor", "cemeteries of london", "lost!",
                                                               "42", "lovers in japan", "yes", "viva la vida",
                                                               "violet hill", "strawberry swing",
                                                               "death and all his friends"],
                "Mylo Xyloto" : ["mylo xyloto", "hurts like heaven", "paradise", "charlie brown",
                                 "us against the world", "m.m.i.x", "every teardrop is a waterfall", "major minus",
                                 "u.f.o", "princess of china", "up in flames", "a hopeful transmission",
                                 "don't let it break your heart", "up with the birds"],
                "Highway To Hell" : ["highway to hell", "girls got rhythm", "walk all over you", "touch too much",
                                     "beating around the bush", "shot down in flames", "get it hot",
                                     "if you want blood (you\'ve got it)", "love hungry man", "night prowler"],
                "The Foundation" : ["toes", "whatever it is", "jolene", "where the boat leaves from", "it's not ok",
                                    "free", "chicken fried", "mary", "different kind of fine", "highway 20 ride",
                                    "sic 'em on a chicken"],
                "If You're Reading This It's Too Late" : ["legend", "energy", "10 bands", "know yourself", "no tellin\'",
                                                          "madonna", "6 god", "star67", "preach",
                                                          "wednesday night interlude", "used to", "6 man",
                                                          "now & forever", "company", "you & the 6", "jungle",
                                                          "6pm in new york"],
                "Random Access Memories" : ["give life back to music", "the game of love", "giorgio by moroder",
                                            "within", "instant crush", "lose yourself to dance", "touch",
                                            "get lucky", "beyond", "motherboard", "fragments of time", "doin' it right",
                                            "contact"]
               }
    thirty_percent_map = {}

    track_info_map = {}
    song_count = 0

    # Loop through albums and songs and get audio_summary data and add to track_info map.
    for album in album_artist_map.keys():
        artist_name = album_artist_map[album]
        for song_title in album_song_map[album]:
            song_count += 1
            if song_count == 100:
                config.ECHO_NEST_API_KEY="6YTN6GQSWUG4JBD0B"
            results = song.search(artist=artist_name,title=song_title)
            track_info_map[song_title] = results[0].audio_summary
            print song_title

    # Write data to a json file
    file_name = "svsm_trackinfomap.json"
    json.dump(track_info_map,open(file_name,"wb"))
    print file_name + " json file written."

    file_name = "svsm_albumartistmap.json"
    json.dump(album_artist_map,open(file_name,"wb"))
    print file_name + " json file written."

    file_name = "svsm_albumsongmap.json"
    json.dump(album_song_map,open(file_name,"wb"))
    print file_name + " json file written."

if __name__ == "__main__":
    main()
