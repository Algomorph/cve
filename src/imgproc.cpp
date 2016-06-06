//  ================================================================
//  Created by Gregory Kramida on 4/29/16.
//  Copyright (c) 2016 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#include <imgproc/imgproc.hpp>

namespace cve{

    cv::Mat non_max_suppression(cv::Mat inputMatrix, int neighborhood_size) {

        for(int block_row = 0; block_row < inputMatrix.rows - neighborhood_size; block_row++){
            for(int block_col = 0; block_col < inputMatrix.cols - neighborhood_size; block_col++){
                //TODO
                //for(int cur_row = block_row; cur_row < )
            }
        }

        return inputMatrix;
    }
}
