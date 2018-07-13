// reduction kernel finding the max value of the local data
__kernel void reduce_max(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int groupNum = get_group_id(0);

	//cache the data from global to local.
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// compare all local values and keep highest one
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid] < scratch[lid + i])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// determin which value is larger, the current largest or the new contender.
	if (!lid) {
		if (scratch[lid] > B[groupNum])
			B[groupNum] = scratch[lid];	//since the contender is larger, move it to be the new largest
	}
}
// reduction kernel finding the min value of the local data
__kernel void reduce_min(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int groupNum = get_group_id(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// compare all local values and keep lowest one
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if (scratch[lid] > scratch[lid + i])
				scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//just a precaustion . really.
	if (lid == 0) {
		if (scratch[lid] < B[groupNum])
			B[groupNum] = scratch[lid];
	}
}
// reduction kernel finding the sum value of the local data
__kernel void reduce_sum(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int groupNum = get_group_id(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	// sum the local values
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// set the local sum to the equivilant global output depositry for this workgroup
	if (!lid) {
		B[groupNum] = scratch[lid];
	}
}

//__kernel void hist_atomic(__global const int* A, __global int* BinData, __local int* H) {
//	int id = get_global_id(0);
//	int lid = get_local_id(0);
//	//clear the scratch bins
//	if (lid < BinData[0])	// BinData[0] = nr_bins  
//		H[lid] = 0;
//	barrier(CLK_LOCAL_MEM_FENCE);
//	atomic_inc(&H[bin_index(A[id])]);
//}

// sort into assending order.
__kernel void selection_sort_local_float(__global const float *A, __global float *B, __local float *scratch)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int LN = get_local_size(0);
	int blocksize = LN;

	float ikey = A[id];

	int pos = 0;
	for (int j = 0; j < N; j += blocksize)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int index = get_local_id(0); index<blocksize; index += LN)
		{
			scratch[index] = A[j + index];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int index = 0; index<blocksize; index++)
		{
			float jkey = scratch[index];
			bool smaller = (jkey < ikey) || (jkey == ikey && (j + index) < id);
			pos += (smaller) ? 1 : 0;
		}
	}
	B[pos] = ikey;
}

// reduction kenerl to create histogram output.
__kernel void selection_search_local_float(__global const float* A, __global int* E, __local float* scratch, __local int* binScratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int groupNum = get_group_id(0); //  get group id
	//int a =1, b=0, c=0, d=0;	// Bins

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i++) {	//goes though each local value, and detmins where it sits in each bin
			//printf("lid: %d\n", lid);
			if ((scratch[i] > -30) && (scratch[i] <= -10))//ranges > -30 < -10 | > -10 < 10 | >10 <30 | >30 <50
			{
				//B[(groupNum*4)]= B[(groupNum * 4)] +1;
				binScratch[0]++;
			}
			else if ((scratch[i] > -10) && (scratch[i] <= 10))
			{
				binScratch[1]++;
			}
			else if ((scratch[i] > 10) && (scratch[i] <= 30))
			{
				binScratch[2]++;
			}
			else if ((scratch[i] > 30) && (scratch[i] <= 50))
			{
				binScratch[3]++;
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	E[(groupNum*4)]= binScratch[0];	//output the local bins to the equivilant bins for the work group.
	E[(groupNum * 4)+1] = binScratch[1];
	E[(groupNum * 4)+2] = binScratch[2];
	E[(groupNum * 4)+3] = binScratch[3];
}