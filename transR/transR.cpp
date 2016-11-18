#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>

using namespace std;

const float pi = 3.141592653589793238462643383;

int transRThreads = 8;
int transRTrainTimes = 0;
int nbatches = 100;
int dimension = 100;
int dimensionR = 100;
float transRAlpha = 0.001;
float margin = 1;

string inPath = "../data/";
string outPath = "../data/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};

struct cmp_list {
	int minimal(int a,int b) {
		if (a > b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) > minimal(b.h, b.t));
	}
};

Triple *trainHead, *trainTail, *trainList;

/*
	There are some math functions for the program initialization.
*/
unsigned long long *next_random;

unsigned long long randd(int id) {
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res<0)
		res+=x;
	return res;
}

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

void norm(float *con, int dimension) {
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(float *con, float *matrix) {
	float tmp, x;
	int last;
	x = 0;
	last = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp = 0;
		for (int jj=0; jj < dimension; jj++) {
			tmp += matrix[last] * con[jj];
			last++;
		}
		x += tmp * tmp;
	}
	if (x>1) {
		float lambda = 1;
		last = 0;
		for (int ii = 0; ii < dimensionR; ii++) {
			tmp = 0;
			for (int jj = 0; jj < dimension; jj++) {
				tmp += matrix[last] * con[jj];
				last++;
			}
			tmp = tmp + tmp;
			last -= dimension;
			for (int jj = 0; jj < dimension; jj++) {
				matrix[last] -= transRAlpha * lambda * tmp * con[jj];
				con[jj] -= transRAlpha * lambda * tmp * matrix[last];
				last++;
			}
			last += dimension;
		}
	}
}

int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *matrix;
float *relationVecDao, *entityVecDao, *matrixDao;

void norm(int h, int t, int r, int j) {
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		// norm(entityVecDao + dimension * h, matrixDao + dimension * dimensionR * r);
		// norm(entityVecDao + dimension * t, matrixDao + dimension * dimensionR * r);
		// norm(entityVecDao + dimension * j, matrixDao + dimension * dimensionR * r);
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimensionR, sizeof(float));
	freqRel = (int *)calloc(relationTotal, sizeof(int));
	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii < dimensionR; ii++)
			relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
	}

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	entityVec = (float *)calloc(entityTotal * dimension, sizeof(float));
	freqEnt = (int *)calloc(entityTotal, sizeof(int));
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i * dimension, dimension);
	}

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(tripleTotal, sizeof(Triple));
	trainList = (Triple *)calloc(tripleTotal, sizeof(Triple));
	tripleTotal = 0;
	while (fscanf(fin, "%d", &trainList[tripleTotal].h) == 1) {
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].t);
		tmp = fscanf(fin, "%d", &trainList[tripleTotal].r);

		freqEnt[trainList[tripleTotal].t]++;
		freqEnt[trainList[tripleTotal].h]++;
		freqRel[trainList[tripleTotal].r]++;

		trainHead[tripleTotal].h = trainList[tripleTotal].h;
		trainHead[tripleTotal].t = trainList[tripleTotal].t;
		trainHead[tripleTotal].r = trainList[tripleTotal].r;

		trainTail[tripleTotal].h = trainList[tripleTotal].h;
		trainTail[tripleTotal].t = trainList[tripleTotal].t;
		trainTail[tripleTotal].r = trainList[tripleTotal].r;

		tripleTotal++;
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());
	sort(trainTail, trainTail + tripleTotal, cmp_tail());
	sort(trainList, trainList + tripleTotal, cmp_list());

	lefHead = (int *)calloc(entityTotal, sizeof(int));
	rigHead = (int *)calloc(entityTotal, sizeof(int));
	lefTail = (int *)calloc(entityTotal, sizeof(int));
	rigTail = (int *)calloc(entityTotal, sizeof(int));
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);
	for (int i = 1; i < tripleTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (float *)calloc(relationTotal,sizeof(float));
	right_mean = (float *)calloc(relationTotal,sizeof(float));
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}

	matrix = (float *)calloc(relationTotal * dimension * dimensionR, sizeof(float));
	entityVecDao = (float *)calloc(dimension * entityTotal, sizeof(float));
	relationVecDao = (float *)calloc(dimensionR * relationTotal, sizeof(float));
	matrixDao = (float *)calloc(dimension * dimensionR * relationTotal, sizeof(float));


	for (int i = 0; i < relationTotal; i++)
		for (int j = 0; j < dimensionR; j++)
			for (int k = 0; k < dimension; k++)
				if (j == k)
					matrix[i * dimension * dimensionR + j * dimension + k] = 1;
				else
					matrix[i * dimension * dimensionR + j * dimension + k] = 0;

	    // fin = fopen("A.bern", "r");
	    // for (int i = 0; i < relationTotal; i++)
	    //         for (int jj = 0; jj < dimension; jj++)
	    //             for (int ii = 0; ii < dimensionR; ii++)
	    //                 tmp = fscanf(fin, "%f", &matrix[i * dimensionR * dimension + jj + ii * dimension]);
	    // fclose(fin);
	    
        FILE* f1 = fopen("../transRdata/entity2vec.bern","r");
        for (int i=0; i<entityTotal; i++)
        {
            for (int ii=0; ii< dimension; ii++)
                tmp = fscanf(f1,"%f",&entityVec[i * dimension + ii]);
            norm(entityVec + i * dimension, dimension);
        }
        fclose(f1);

        FILE* f2 = fopen("../transRdata/relation2vec.bern","r");
        for (int i=0; i<relationTotal; i++)
        {
            for (int ii=0; ii<dimension; ii++)
                tmp = fscanf(f2,"%f",&relationVec[i * dimensionR + ii]);
        }
        fclose(f2);

}

/*
	Training process of transR.
*/

int transRLen;
int transRBatch;
float res;

float calc_sum(int e1, int e2, int rel) {
	int lastM = rel * dimension * dimensionR;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastr = rel * dimensionR;
	float sum = 0, tmp1, tmp2;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp1 = 0;
		tmp2 = 0;
		for (int jj = 0; jj < dimension; jj++) {
			tmp1 += matrix[lastM] * entityVec[last1 + jj];
			tmp2 += matrix[lastM] * entityVec[last2 + jj];
			lastM++;
		}
		sum += fabs(tmp1 + relationVec[lastr + ii] - tmp2);
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, int same) {
	int lasta1 = e1_a * dimension;
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	float x;
	for (int ii=0; ii < dimensionR; ii++) {
		x = 0;
		for (int jj = 0; jj < dimension; jj++) {
			x -= matrix[lastM + jj] * entityVec[lasta1 + jj];
			x += matrix[lastM + jj] * entityVec[lasta2 + jj];
		}
		x = x - relationVec[lastar + ii];
		if (x > 0)
			x = belta * transRAlpha;
		else
			x = -belta * transRAlpha;
		for (int jj = 0; jj < dimension; jj++) {
			matrixDao[lastM + jj] -=  x * (entityVec[lasta1 + jj] - entityVec[lasta2 + jj]);
			entityVecDao[lasta1 + jj] -= x * matrix[lastM + jj];
			entityVecDao[lasta2 + jj] += x * matrix[lastM + jj];
		}
		relationVecDao[lastar + ii] -= same * x;
		lastM = lastM + dimension;
	}
}

void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a);
	float sum2 = calc_sum(e1_b, e2_b, rel_b);
	if (sum1 + margin > sum2) {
		res += margin + sum1 - sum2;
		gradient(e1_a,e2_a,rel_a,-1,1);
    	gradient(e1_b,e2_b,rel_b,1,1);
	}
}

int corrupt_head(int id, int h, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

void* transRtrainMode(void *con) {
	int id;
	id = (unsigned long long)(con);
	next_random[id] = rand();
	for (int k = transRBatch / transRThreads; k >= 0; k--) {
		int j;
		int i = rand_max(id, transRLen);
		float pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		// int pr = 500;
		if (randd(id) % 1000 < pr) {
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j);
	}
}

void* train_transR(void *con) {
	transRLen = tripleTotal;
	transRBatch = transRLen / nbatches;
	next_random = (unsigned long long *)calloc(transRThreads, sizeof(unsigned long long));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(matrixDao, matrix, dimension * relationTotal * dimensionR * sizeof(float));
	for (int epoch = 0; epoch < transRTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(transRThreads * sizeof(pthread_t));
			for (int a = 0; a < transRThreads; a++)
				pthread_create(&pt[a], NULL, transRtrainMode,  (void*)a);
			for (int a = 0; a < transRThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(matrix, matrixDao, dimension * relationTotal * dimensionR * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of transR.
*/

void out_transR() {
		FILE* f2 = fopen("relation2vec.bern", "w");
		FILE* f3 = fopen("entity2vec.bern", "w");
		for (int i=0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen("A.bern","w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrix[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f1);
}

/*
	Main function
*/

int main() {
	init();
	train_transR(NULL);
	out_transR();
	return 0;
}
