#include <iostream>
#include <vector>
using namespace std;

class Solution
{
public:
    vector<int>mult(vector<int>&nums)
    {
        vector<int>output(nums.size(),0);
        vector<int>left(nums.size(),1);
        vector<int>right(nums.size(),1);
        for(int i = 1;i< nums.size();++i)
        {
            left[i] = left[i-1]*nums[i-1];
        }
        for(int i = nums.size()-2;i>=0;--i)
        {
            right[i]=right[i+1]*nums[i+1];
        }
        for(int i = 0;i<nums.size();++i)
        {
            output[i] = left[i]*right[i];
        }
        return output;
    }
}
int main()
{

}




vector<int>test = {1,2,3,4}
vector<int>output = {24,12,8,6};

vector<>test2 = {-1,1,0,-3};
output2 = {0,0,9,0};




