from xlrd import open_workbook
import numpy as np

__author__ = 'pegah'

class manage_AS:

    def get_CS(self, AS):
        CSs = []
        with open('../Files/WS-AS-normalize.txt', 'r') as file:
            for line in file:
                lines = line.split(" ")
                if (lines[0] == AS) and int(lines[1]) < 4500:
                    CSs.append(lines[1])
        return CSs

    def get_AS_list(self):
        list = []

        count = 0
        with open("../Files/AAS.txt", 'r') as file:
            for line in file:
                if (count > 0):
                    row = line.split()
                    if row[0] not in list:
                        list.append(row[0])

                count += 1

        return list

    def merge_text(self, file1, file2):

        text_file = open("tp-rt-merge.txt", "w")

        with open(file1) as f1, open(file2) as f2:
            for x, y in zip(f1, f2):
                #text_file.write("{0} {1} {2}\n".format(x.strip()[0:2], x.strip()[3], ((y.strip()).split())[3]))
                text_file.write("{0} {1}\n".format(x.strip(), ((y.strip()).split())[3]))
        return

    def get_qos_old(self, service_ID, time_slice_ID):
        output = []
        file = "tp-rt-merge.txt"
        with open(file) as qos:
            for line in qos:
                words_list = line.split()
                if words_list[1] == service_ID and words_list[2] == time_slice_ID:
                    #response time and Throughput
                    QoS = words_list[3:5]
                    if QoS not in output:
                        output.append([np.float32(i) for i in QoS])
                    #output.append(words_list[3:5])

        return list(output)

    def get_qos(self, service_ID, time_slice_ID, matrix):

        output = []
        get_index = []
        counter = 0

        while counter < matrix.shape[0]:
            if (int(matrix[counter, :][2]) < time_slice_ID) and (int(matrix[counter, :][1]) > service_ID):
                break
            else:
                words_list = matrix[counter, :]

                if int(words_list[1]) == service_ID and int(words_list[2]) == time_slice_ID:
                        QoS = words_list[3:5]
                        QoS = [i for i in QoS]
                        get_index.append(counter)
                        if QoS not in output:
                            output.append([i for i in QoS])
            counter += 1


        matrix_out = np.delete(matrix, get_index, axis=0)
        #print 'rows', matrix_out.shape[0]

        return (list(output), matrix_out)


