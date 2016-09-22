/**
 * Author: Sudnya Diamos <mailsudnya@gmail.com>
 * Rabin Karp algorithm to find whether test string contains pattern as substring
**/

#include <iostream>
#include <cmath>
class RabinKarp
{
    public:
        RabinKarp(const std::string& testStr, const std::string& pattern)
        : _testText(testStr), _searchPattern(pattern)
        {
        }


    public:
        int isSubstringAt()
        {
            int retVal = -1;
            
            int testTextSize = _testText.size();
            int patternSize = _searchPattern.size();
            
            int patternHash = getHash(_searchPattern, 0, patternSize);
            std::cout << "\nPattern Hash: " << patternHash;
            int testHash = getHash(_testText, 0, patternSize);


            for (int i = 1; i <= testTextSize - patternSize+1; ++i)
            {
                std::cout << "\nTest Hash: " << testHash;
                if (patternHash == testHash)
                {
                    if (isEqual(_testText, i-1, i+patternSize-2, _searchPattern, 0, patternSize-1))
                    {
                        std::cout << "\nMatch found\n";
                        return i-1;
                    }
                }
                if (i < testTextSize - patternSize + 1)
                {
                    testHash = getRollingHash(_testText, i-1, i+patternSize-1, testHash, patternSize);
                }

            }

            return -1;
        }
    
    private:
        int getHash(std::string pattern, int start, int end)
        {
            int retVal = 0;
            for (int i = 0; i < end; ++i)
            {
                retVal += (int)pattern[i] * (pow(HASH_PRIME, i));
            }
            return retVal;
        }

        int getRollingHash(std::string str, int oldIdx, int newIdx, int oldHash, int patternLength)
        {
            std::cout << "\nOld hash: " << oldHash;
            int newHash = oldHash - (int)str[oldIdx];
            newHash /= HASH_PRIME;
            newHash += (int)str[newIdx] * (pow(HASH_PRIME, patternLength-1));
            std::cout << "\nNew hash: " << newHash;
            return newHash;
        }

        bool isEqual(std::string test, int start, int end, std::string pattern, int start2, int end2)
        {
            /*int t = start;
            int t2 = start2;
            while (t <= end && t2 <= end2)
            {
                std::cout << "\n" << test[t] << " == ? " << pattern[t2] << "\n";
                t++;
                t2++;
            }*/
            if (end - start != end2 - start2)
            {
                //std::cout << "diff start end dont match\n" << start << ", " << end << " vs " << start2 << " , " << end2;
                return false;
            }
            while (start <= end && start2 <= end2)
            {
                if (test[start] != pattern[start2])
                    return false;
                start++;
                start2++;
            }
            return true;
        }
    private:
        std::string _testText;
        std::string _searchPattern;
        const int HASH_PRIME = 3;
};

int main()
{
    std::string pattern = "ABAC";
    std::string test = "ABABBABACABAA";

    RabinKarp rk(test, pattern);
    int match = rk.isSubstringAt();
    if (match == -1)
    {
        std::cout << "\nPattern " << pattern << " is not contained in " << test << std::endl;
    } 
    else
    {
        std::cout << "\nPattern " << pattern << " is substring of " << test << " at offset: " << match << std::endl;
    }
    return 0;
}
