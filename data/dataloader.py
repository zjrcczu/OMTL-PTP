import os, random, numpy as np, copy

from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log


class data_generator(object):

    def __init__(self, parser, log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'
        if parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:#{'eth', 'hotel', 'univ', 'zara1', 'zara2'}
            data_root = parser.data_root_ethucy            
            seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root


        # print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load:
            # print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, parser, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames - 1) * self.frame_skip - parser.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
        
        self.sample_list = list(range(self.num_total_samples))
        if parser.shuffle:
            self.shuffle()
        #self.sample_list =[177, 406, 587, 835, 573, 391, 568, 526, 945, 946, 734, 720, 659, 494, 749, 181, 689, 313, 222, 112, 293, 74, 487, 784, 77, 42, 453, 594, 215, 301, 638, 923, 15, 47, 421, 282, 440, 378, 837, 883, 632, 894, 153, 502, 263, 561, 908, 617, 824, 919, 6, 722, 675, 450, 875, 879, 78, 483, 873, 105, 305, 599, 110, 100, 321, 939, 231, 693, 385, 138, 147, 896, 914, 314, 0, 7, 362, 355, 214, 455, 746, 102, 103, 174, 241, 388, 555, 97, 171, 402, 905, 858, 491, 529, 83, 793, 432, 286, 359, 546, 530, 578, 57, 331, 566, 552, 246, 444, 431, 233, 918, 535, 185, 70, 164, 680, 684, 694, 409, 527, 503, 855, 628, 350, 522, 622, 601, 342, 940, 657, 348, 730, 778, 528, 115, 626, 86, 221, 572, 206, 910, 696, 631, 280, 834, 297, 800, 365, 28, 844, 119, 770, 417, 196, 167, 506, 591, 705, 884, 866, 801, 466, 741, 169, 504, 668, 782, 31, 130, 308, 290, 142, 50, 344, 531, 339, 678, 465, 532, 610, 408, 791, 701, 726, 166, 317, 34, 310, 419, 412, 643, 836, 117, 323, 315, 157, 795, 650, 24, 335, 158, 247, 922, 416, 913, 598, 262, 604, 218, 356, 351, 931, 571, 141, 379, 198, 869, 55, 258, 929, 714, 501, 311, 418, 107, 273, 708, 783, 49, 718, 948, 451, 845, 485, 189, 570, 76, 253, 764, 590, 707, 14, 645, 322, 186, 120, 551, 635, 360, 692, 787, 270, 841, 577, 560, 639, 520, 892, 563, 265, 636, 549, 209, 619, 554, 461, 656, 912, 244, 832, 733, 340, 687, 612, 373, 508, 349, 652, 690, 175, 771, 8, 909, 889, 154, 79, 806, 540, 785, 11, 585, 165, 318, 826, 445, 239, 201, 283, 938, 606, 234, 804, 825, 629, 425, 872, 73, 660, 80, 104, 518, 227, 797, 333, 132, 490, 893, 23, 500, 188, 472, 886, 842, 802, 430, 200, 170, 640, 686, 944, 661, 473, 470, 413, 559, 111, 82, 890, 467, 40, 865, 149, 424, 511, 805, 309, 848, 557, 820, 794, 755, 426, 863, 156, 296, 337, 790, 319, 904, 67, 582, 84, 562, 765, 548, 862, 411, 277, 39, 565, 860, 799, 703, 161, 292, 375, 307, 392, 876, 93, 618, 420, 19, 394, 17, 614, 536, 762, 401, 569, 3, 856, 69, 492, 475, 266, 405, 137, 441, 719, 497, 773, 928, 25, 942, 580, 410, 644, 673, 667, 754, 85, 415, 828, 368, 240, 927, 363, 436, 46, 197, 372, 140, 274, 245, 592, 723, 537, 881, 752, 947, 129, 627, 135, 581, 766, 259, 53, 397, 843, 225, 90, 109, 833, 438, 173, 831, 243, 22, 937, 217, 468, 116, 683, 9, 663, 477, 609, 92, 915, 934, 66, 564, 369, 809, 346, 108, 748, 789, 745, 454, 666, 509, 338, 95, 655, 891, 810, 400, 371, 289, 936, 933, 776, 65, 68, 204, 505, 682, 699, 709, 261, 91, 56, 623, 249, 336, 238, 917, 737, 26, 662, 646, 134, 732, 887, 13, 674, 486, 36, 38, 366, 751, 254, 127, 347, 484, 446, 780, 493, 758, 808, 823, 943, 81, 352, 920, 676, 553, 281, 574, 556, 651, 327, 10, 205, 469, 144, 44, 133, 935, 242, 533, 649, 633, 822, 542, 516, 203, 452, 756, 361, 343, 819, 59, 545, 898, 474, 916, 870, 404, 899, 567, 798, 462, 264, 332, 383, 859, 387, 101, 123, 448, 695, 60, 515, 727, 276, 236, 288, 163, 298, 613, 316, 380, 58, 901, 679, 443, 757, 786, 180, 818, 724, 704, 184, 1, 437, 710, 267, 389, 813, 464, 544, 367, 630, 538, 324, 407, 396, 807, 846, 588, 257, 769, 760, 543, 579, 880, 711, 118, 251, 125, 911, 235, 151, 781, 295, 864, 479, 126, 589, 716, 287, 122, 698, 150, 271, 159, 399, 113, 376, 269, 596, 600, 187, 713, 358, 607, 256, 747, 902, 608, 341, 926, 742, 183, 61, 52, 429, 106, 357, 759, 229, 731, 512, 744, 586, 195, 524, 850, 900, 330, 4, 725, 168, 669, 353, 817, 728, 510, 94, 595, 312, 878, 763, 871, 131, 96, 932, 143, 146, 71, 395, 210, 715, 706, 558, 192, 941, 442, 792, 87, 88, 41, 433, 252, 121, 885, 849, 637, 381, 907, 495, 176, 665, 625, 482, 583, 427, 775, 37, 334, 906, 35, 213, 33, 480, 20, 230, 620, 498, 603, 664, 178, 62, 299, 772, 738, 386, 284, 584, 750, 525, 513, 325, 328, 320, 398, 250, 5, 278, 615, 63, 903, 576, 422, 285, 539, 672, 148, 774, 32, 291, 99, 345, 854, 403, 642, 521, 27, 811, 514, 447, 653, 152, 821, 51, 75, 224, 830, 517, 735, 499, 489, 435, 481, 194, 796, 449, 685, 370, 89, 488, 736, 519, 739, 300, 226, 829, 12, 677, 272, 216, 688, 840, 155, 260, 605, 534, 223, 753, 895, 428, 191, 471, 671, 172, 48, 463, 712, 18, 803, 382, 374, 21, 611, 114, 160, 228, 326, 575, 496, 670, 921, 294, 193, 857, 456, 868, 839, 220, 702, 439, 507, 547, 616, 329, 423, 874, 390, 624, 523, 779, 182, 275, 45, 847, 717, 634, 681, 145, 721, 700, 930, 888, 697, 852, 72, 924, 658, 306, 54, 414, 207, 208, 29, 457, 98, 459, 460, 827, 232, 162, 237, 179, 851, 853, 654, 597, 212, 384, 814, 202, 393, 30, 268, 199, 219, 303, 248, 812, 550, 621, 304, 190, 648, 458, 2, 897, 364, 777, 768, 691, 476, 925, 354, 816, 16, 641, 877, 302, 767, 255, 478, 788, 743, 729, 136, 43, 541, 838, 128, 647, 64, 602, 434, 593, 740, 861, 124, 211, 815, 279, 377, 867, 139, 761, 882]



        print(self.sample_list)


        self.index = 0
        # print_log(f'total num samples: {self.num_total_samples}', log)
        # print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):

        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):

        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()
