/*
 * MotifUtils.h
 *
 *  Created on: Dec 2, 2018
 *
 */

#ifndef INCLUDES_MOTIFUTILS_H_
#define INCLUDES_MOTIFUTILS_H_

#include <stdint.h>
#include <vector>
using namespace std;
typedef int64_t int64;

vector<vector<int> *> *permutations(vector<int> &nbunch);
vector<vector<unsigned int> *> *permutations(vector<unsigned int> &nbunch);
vector<vector<int> *> *combinations(vector<int> &nbunch);
vector<vector<unsigned int> *> *neighbors_combinations(
		const unsigned int* neighbors, int64 start, int64 end, int deg=2);
vector<vector<unsigned int> *> *combinations(vector<unsigned int> &nbunch);
int bool_vector_to_int(vector<bool> &edges);


#endif /* INCLUDES_MOTIFUTILS_H_ */

