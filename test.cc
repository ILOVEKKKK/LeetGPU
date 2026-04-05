#include <iostream>
#include <vector>
#include <string>
using namespace std;
class Solution
{
public:
    float angle(string &time)
    {
        int hour = 0;
        int minute = 0;
        float result = 0.0f;

        for(int i = 0;i < time.size();++i)
        {
            if(time[i] == ':')
            {
                hour = stoi(time.substr(0,i));
                minute = stoi(time.substr(i+1,time.size()-i-1));
                break;
            }
        }

        float angle_minute = 360.0f*(minute/60.0f);
        float angle_hour = 360.0f*((hour%12)/12.0f)+30*(minute/60.0f);

        result = abs(angle_hour-angle_minute) > 180?360.0f-abs(angle_hour-angle_minute):abs(angle_hour-angle_minute);

        return result;
    }

};