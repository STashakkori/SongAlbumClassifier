__author__ = 'campuser'

import json
import xlrd
import csv
import os
import numpy
from matplotlib.patches import Ellipse
import matplotlib.colors as col
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from xlsxwriter.workbook import Workbook
from sklearn import mixture

"""
    main :: method that loads a json document from disk to local variables.
"""
def main():
    debug_mode = False

    file_name = "svsm_trackinfomap.json"
    track_info_map = json.loads(open(file_name,"rb").read())

    file_name = "svsm_albumartistmap.json"
    album_artist_map = json.loads(open(file_name,"rb").read())

    file_name = "svsm_albumsongmap.json"

    album_song_map = json.loads(open(file_name,"rb").read())

    if debug_mode:
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        print len(track_info_map.keys())
        pp.pprint(track_info_map)

    important_info, song_names = map_to_matrix("finalproject.csv",track_info_map)
    sorted_matrix, album_lengths = color_songs(important_info, song_names, album_song_map)
    measures = []
    same_measures = []
    for i in xrange(len(important_info[0])) :
        for j in xrange(i+1,len(important_info[0])) :
            similarity_measure_k = k_means_maker(important_info, [i,j], album_song_map, song_names)
            similarity_measure_g = gaussian_mixture_model(important_info, [i,j], album_song_map, song_names)
            measures.append(((similarity_measure_k + similarity_measure_g) / 2, i, j))
    for i in xrange(len(important_info[0])) :
        similarity_measure_k = k_means_maker(important_info, [i,i], album_song_map, song_names)
        similarity_measure_g = gaussian_mixture_model(important_info, [i,i], album_song_map, song_names)
        same_measures.append(((similarity_measure_k + similarity_measure_g) / 2))
    sorted_measures = numpy.asarray(measures)
    sorted_measures = sorted_measures[numpy.argsort(sorted_measures[:, 0])]
    print sorted_measures
    visualizer(sorted_matrix, 1, 3, album_lengths)


"""
    write_datastructures
"""
def map_to_matrix(file_name, track_info_map):
    headers =["","acousticness","danceability","duration","energy","instrumentalness","key","liveness","loudness","mode",
              "speechiness","tempo","time_signature","valence"]

    song_count = 0
    columns = len(headers)
    for song_name in track_info_map.keys():
        values = []
        song_count += 1
        for header in headers:
            if header in track_info_map[song_name]:
                 values.append(track_info_map[song_name][header])

        values.insert(0,song_name)
    data_matrix = numpy.empty((song_count, columns)).astype(str)

    with open(file_name,'rb') as f:
        reader = csv.reader(f)
        for rindex, row in enumerate(reader):
            for cindex, column in enumerate(row):
                data_matrix[rindex-1, cindex-1] = column
    return data_matrix[:, :-1].astype(float), data_matrix[:, -1].astype(str)

def map_to_csv(file_name, track_info_map):
    headers =["","acousticness","danceability","duration","energy","instrumentalness","key","liveness","loudness","mode",
              "speechiness","tempo","time_signature","valence"]

    csv_file = open(file_name,"wb")
    writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(headers)
    song_count = 0
    for song_name in track_info_map.keys():
        values = []
        song_count += 1
        for header in headers:
            if header in track_info_map[song_name]:
                 values.append(track_info_map[song_name][header])

        values.insert(0,song_name)
        writer.writerow(values)
    csv_file.close()

    try:
        work_book = xlrd.open_workbook("finalproject.xls")
        work_sheet = work_book.get_sheet('datasheet1')

    except:
        base_name = os.path.splitext(file_name)[0]
        work_book = Workbook(base_name + ".xlsx")
        work_sheet = work_book.add_worksheet()

    with open(file_name,'rb') as f:
        reader = csv.reader(f)
        for rindex, row in enumerate(reader):
            for cindex, column in enumerate(row):
                work_sheet.write(rindex,cindex,column)
    work_book.close()


def k_means_maker(data, dimensions, map, song_names):
    numpy.random.seed(1)
    kmeans = KMeans(int(len(data)/10))
    kmeans.fit(data[:,dimensions].reshape((len(data), 2)))
    clusters = kmeans.predict(data[:,dimensions].reshape((len(data), 2)))
    something = compute_similarity_map(clusters, map, song_names)
    return something

def compute_similarity_map (matrix, map, song_names) :
    a = numpy.empty((len(matrix), len(matrix)))
    b = numpy.empty((len(matrix), len(matrix)))
    for x in xrange(len(matrix)) :
        for y in xrange(len(matrix)) :
            if matrix[x] == matrix[y] :
                a[x,y] = 1
            else :
                a[x,y] = 0
            if album_finder(map, song_names[x]) == album_finder(map, song_names[y]) :
                b[x,y] = 1
            else :
                b[x,y] = 0
    return similarity_measurer(a,b)

def similarity_measurer(a, b):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    if a.shape != b.shape:
        print "Bad input dude!"
        exit(1)

    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            if a[i ,j] and b[i ,j]:
               true_positives += 1

            elif not a[i ,j] and not b[i ,j]:
                true_negatives += 1

            elif not a[i ,j] and b[i ,j]:
                false_positives += 1

            elif a[i ,j] and not b[i ,j]:
                false_negatives += 1

    total_elements = a.shape[0] * a.shape[1]

    # return the number of accurate readings
    return float(true_positives + true_negatives) / total_elements

def gaussian_mixture_model(data, number_of_dimensions, map, song_names):
    g = mixture.GMM(n_components = int(len(data)/10))
    g.fit(data[:,number_of_dimensions].reshape((len(data), 2)))
    prediction = g.predict(data[:,number_of_dimensions].reshape((len(data), 2)))
    something = compute_similarity_map(prediction, map, song_names)
    return something

def extract_train_and_test(data):
    number_of_songs = len(data)
    seventy_percent = int(.70 * number_of_songs)
    return data[0:seventy_percent], data[seventy_percent:]

def color_songs(data, song_names, map) :
    albums = []
    for i in song_names:
        print i
        albums.append(album_finder(map,i))
    sorted_matrix = numpy.empty((len(data), 14))
    sorted_matrix[:,:13] = data
    sorted_matrix = sorted_matrix.astype(str)
    sorted_matrix[:,13] = numpy.asarray(albums).astype(str)
    sorted_matrix = sorted_matrix[numpy.argsort(sorted_matrix[:,13])]
    count = 1
    lengths = []
    for i in range(1, len(sorted_matrix)):
        if sorted_matrix[i, 13] == sorted_matrix[i-1, 13]:
            count += 1
        else:
            lengths.append(count)
            count = 1
    lengths.append(count)
    return sorted_matrix[:, :13].astype(float), lengths


def visualizer(data, x, y, lengths):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle("Clustered Results")
    colors = plt.get_cmap("gist_rainbow")
    norm = col.Normalize(vmin = 0, vmax = len(lengths))
    color_map = cm.ScalarMappable(cmap = colors, norm = norm)
    index = 0
    for i in range(len(lengths)):
        plt.scatter(data[index:index + lengths[i], x], data[index:index + lengths[i], y], c = color_map.to_rgba(i))
        if lengths[i] != 1:
            cov = numpy.cov(data[index:index + lengths[i], x], data[index:index + lengths[i], y], ddof = 0)
        else:
            cov = numpy.asarray([[0, 0], [0,0]])
        values, vectors = numpy.linalg.eig(cov)
        angle = numpy.arctan((vectors[0,1] - vectors[1,1])/(vectors[0,0] - vectors[1,0]))
        ellipse = Ellipse(xy = (numpy.mean(data[index:index + lengths[i], x]), numpy.mean(data[index:index + lengths[i], y])), width = 2 * 2 * numpy.sqrt(values[0]),
                          height = 2 * 2 * numpy.sqrt(values[1]), angle = numpy.rad2deg(angle))
        ellipse.set_edgecolor(color_map.to_rgba(i))
        ellipse.set_facecolor("none")
        ax.add_artist(ellipse)
        plt.scatter(numpy.mean(data[index:index + lengths[i], x]), numpy.mean(data[index:index + lengths[i], y]), c = color_map.to_rgba(i), marker = 'x')
        index += lengths[i]
    plt.show()
    plt.close()

def album_finder(album_song_map,song):
    for album in album_song_map.keys():
        if song in album_song_map[album]:
            return album


if __name__ == "__main__":
    main()
