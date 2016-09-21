
#include <iostream>

void swap(int* inputA, int i, int j)
{
    int temp = inputA[i];
    inputA[i] = inputA[j];
    inputA[j] = temp;
}

int partition(int* inputA, int start, int end)
{
    //simple choice of pivot -> end, bad if already sorted array
    int pivotVal = inputA[end];
    int pIndex = start;

    for (int i = start; i < end; ++i)
    {
        if (inputA[i] <= pivotVal)
        {
            swap(inputA, pIndex, i);
            pIndex++;
        }
    }
    //swap the highest element so far to right most
    swap(inputA, pIndex, end);

    return pIndex;
}

void quickSort(int* inputA, int start, int end)
{
    //only till one element left in subproblem
    if (start < end)
    {
        //get partitioned left, right subarrays where left 
        //contains less than pivot and right more than pivot
        int pivot = partition(inputA, start, end);
        quickSort(inputA, start, pivot-1);
        quickSort(inputA, pivot+1, end);
    }
}

int main()
{
    int inputA[] = {4, 5, 8, 2, 1, 0, 7, 6, 3 };
    quickSort(inputA, 0, 8);
    for (int i  = 0; i < 9; ++i)
    {
        std::cout << inputA[i] << " , ";
    }
}
