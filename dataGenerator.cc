
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
using namespace std;


int main(int argc, const char * argv[]) {
    
    int numValues;
    if ( ! (istringstream(argv[1]) >> numValues) ) numValues = 0;
    
    int numDim;
    if ( ! (istringstream(argv[2]) >> numDim) ) numDim = 0;
    
    int numLabel;
    if ( ! (istringstream(argv[3]) >> numLabel) ) numLabel = 0;
    
    string textfile;
    textfile = argv[4];
    ofstream values;
    
    random_device rd;
    mt19937 gen(rd());
    
    normal_distribution<> norm(2,1);
    
    values.open(textfile, ios::out | ios::trunc);
    
    if(!values.is_open()){
        cout << "Something went wrong when opening " << textfile << endl;
        exit(0);
    }
    
    values << numValues << " "<<numDim << " " << numLabel <<endl;
    
    double value = 0.0;
    int numWritten = 0;
    
    for(int label = 0; label < numLabel; label++){
        for(int i = 0; i < numValues/numLabel; i++){
            for(int j = 0; j < numDim; j++){
                value = norm(gen) + 4*label;
                values << value << " ";
            }
            numWritten++;
            values <<label << endl;
        }
    }
    
    
    int leftover = numValues - numWritten;
    
    if(leftover > 0){
        for(int i = 0; i < leftover; i++){
            for(int j = 0; j < numDim; j++){
                value = norm(gen) + 4*(numLabel-1);
                values << value << " ";
            }
            values << numLabel-1 << endl;
        }
    }
    
    values.close();
    cout << "Data was generated. See " << textfile << endl;
    
    return 0;
    
}


