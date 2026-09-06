import os
import json

import pytest

from generate.images import Checkpoint, Images, DF
from generate.images.generate import update_index_callback
from utils.datatypes import DFRows

class TestImageCheckpointUpdateIndexes:
    # the file is only updated when the image index (in this case 20)
    # cycles after STEP_FOR_SAVE_CHECKPOINT times
    # for this example, we use 10, so 20 could be save normally.
    # be aware that this variable may change things, so in some other cases, 
    # the these tests may not work correctly
    def test_do_not_update_index_callback(self,checkpoint_path):
        checkpoint = Checkpoint(checkpoint_path)

        assert checkpoint.thread_indexes == []

        for _ in range(4):
            checkpoint.thread_indexes.append(0)

        update_index_callback(2,14,checkpoint)

        assert checkpoint.thread_indexes == [0,0,14,0]

        with open(checkpoint_path, "r")  as file:
            j = json.load(file)
            assert j["thread_indexes"] == []

    def test_update_index_callback(self,checkpoint_path):
        checkpoint = Checkpoint(checkpoint_path)

        assert checkpoint.thread_indexes == []

        for _ in range(4):
            checkpoint.thread_indexes.append(0)


        update_index_callback(2,20,checkpoint)
        assert checkpoint.thread_indexes == [0,0,20,0]

        with open(checkpoint_path, "r")  as file:
            j = json.load(file)
            assert j["thread_indexes"] == [0,0,20,0]



class TestImageGenerationStaticMethods:
    def test_correct_combinations_of_measurements(self):
        combs = Images._get_combinations_of_measurements(4)
        assert combs == [
            (0,), (1,), (2,), (3,),
            (0,1), (0,2), (0,3), 
            (1,2), (1,3), 
            (2,3), 
            (0,1,2), (0,1,3), (0,2,3), (1,2,3),
            (0,1,2,3)
        ]
    
    def test_correct_combinations_of_measurements_1_qubit(self):
        combs = Images._get_combinations_of_measurements(1)
        assert combs == [(0,),]
    
    def test_correct_combinations_of_measurements_2_qubits(self):
        combs = Images._get_combinations_of_measurements(2)
        assert combs == [(0,), (1,), (0,1)]


    def test_fix_dist_gaps(self):
        dist = {2:1.0}
        Images._fix_dist_gaps(dist, 3)

        assert len(dist.keys()) == 8
        for i in range(8):
            assert dist.get(i) == (1.0 if i == 2 else 0.0)


class TestImageGeneration:
    def test_generate_circuit_images(self, images_path):
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)
        meas = im._get_combinations_of_measurements(2)

        thread_indexes = []
        img_indexes = []
        def callback(thread_index:int, img_index:int):
            thread_indexes.append(thread_index)
            img_indexes.append(img_index)
            
        row = im._generate_circuit_images(
            0,
            0,
            meas,
            2,
            4,
            1000,
            callback
            )
        
        assert thread_indexes == [0,0,0]
        assert img_indexes == [0,1,2]
        assert sorted(os.listdir(images_path)) == [ f"{i}.png" for i in range(3)]
        assert len(row) == 3
        assert [r["index"] for r in row] == [0,1,2]
        assert [r["total_meas"] for r in row] == [1,1,2]
        assert [r["measurements"] for r in row] == ['[0]', '[1]','[0, 1]']
        assert [r["file"] for r in row] == [ os.path.join(images_path,f"{i}.png") for i in range(3)]
        assert not any([r["file_size_bytes"] <= 0 for r in row]) 

    def test_current_index_generate_image(self,images_path):
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)
        meas = im._get_combinations_of_measurements(2)

        thread_indexes = []
        img_indexes = []
        def callback(thread_index:int, img_index:int):
            thread_indexes.append(thread_index)
            img_indexes.append(img_index)
            
        row = im._generate_circuit_images(
            0,
            0,
            meas,
            2,
            4,
            1000,
            callback,
            current_index=1
            )
        
        assert thread_indexes == [0,0]
        assert img_indexes == [1,2]
        assert sorted(os.listdir(images_path)) == [ f"{i}.png" for i in range(1,3)]
        assert len(row) == 2
        assert [r["index"] for r in row] == [1,2]
        assert [r["total_meas"] for r in row] == [1,2]
        assert [r["measurements"] for r in row] == ['[1]','[0, 1]']
        assert [r["file"] for r in row] == [ os.path.join(images_path,f"{i}.png") for i in range(1,3)]
        assert not any([r["file_size_bytes"] <= 0 for r in row]) 

    def test_base_index_generate_image(self,images_path):
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)
        meas = im._get_combinations_of_measurements(2)

        thread_indexes = []
        img_indexes = []
        def callback(thread_index:int, img_index:int):
            thread_indexes.append(thread_index)
            img_indexes.append(img_index)
            
        row = im._generate_circuit_images(
            0,
            10,
            meas,
            2,
            4,
            1000,
            callback,
            )
        
        assert thread_indexes == [0,0,0]
        assert img_indexes == [10,11,12]
        assert sorted(os.listdir(images_path)) == [ f"1{i}.png" for i in range(3)]
        assert len(row) == 3
        assert [r["index"] for r in row] == [10,11,12]
        assert [r["total_meas"] for r in row] == [1,1,2]
        assert [r["measurements"] for r in row] == ['[0]','[1]','[0, 1]']
        assert [r["file"] for r in row] == [ os.path.join(images_path,f"1{i}.png") for i in range(3)]
        assert not any([r["file_size_bytes"] <= 0 for r in row]) 

    def test_generate_multiple_circuits(self, images_path,checkpoint_path, df_path): 
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)

        checkpoint = Checkpoint(checkpoint_path)
        df = DF(df_path)

        im.generate_images(2,5,4,1000,df,2,checkpoint)

        assert checkpoint.index == 5
        assert df._current_index == 15 # 5 circuits with 3 different measurements each
        assert sorted(os.listdir(images_path)) == sorted([ f"{i}.png" for i in range(15)])

    def test_generate_multiple_circuits_start_from_checkpoint(self, images_path,checkpoint_path,df_path): 
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)

        df = DF(df_path)

        checkpoint = Checkpoint(checkpoint_path)
        checkpoint.index = 2              # already created 2 circuits (6 images) - 2 threads
        checkpoint.thread_indexes = [0,2] # 6 images plus 2, so 8 was already created
                                          # to reach the total of 15, it need to generate
                                          # 7 new images

        im.generate_images(2,5,4,1000,df,2,checkpoint)

        assert checkpoint.index == 5
        assert df._current_index == 7


        # when checkpoint.index == 0 : 
        #       thread 0: _, _, _
        #       thread 1: _, _, _
        # when checkpoint.index == 2 (0 + 2threads from the first)
        #       thread 0: '6.png', '7.png', '8.png'
        #       thread 1: _      ,   _,     '11.png'
        # when checkpoint.index == 4 (2 + 2threads from the second)
        #      thread 0: '12.png', '13.png', '14.png'
                
        assert sorted(['6.png', '7.png', '8.png', '11.png','12.png','13.png','14.png']) == sorted(os.listdir(images_path))

