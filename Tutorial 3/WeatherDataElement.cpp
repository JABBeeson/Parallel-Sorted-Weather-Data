#include<iostream>
#include<array>
#include<string>
using namespace std;

class WeatherDataElement {
	public:
		string stationName;		//module num 0
		int year;				//module num 1
		int month;				//module num 2
		int day;				//module num 3
		int time;
		//array<float, 2> time;	//module num 4
		float temperature;		//module num 5

	void Instantiate(string line) {	

		int dataColumn = 0;
		//int dataRow = 0;
		string temp;
		line += ' ';
		//WeatherDataElement* temp;
		for (int i = 0; i < line.length(); i++)
		{
			if (line.at(i) == ' ' )
			{
				switch (dataColumn)
				{
				case 0:
					//stationName = temp; 
					break;
				case 1:
					//year = stoi(temp);
					break;
				case 2:
					month = stoi(temp);
					break;
				case 3:
					//day = stoi(temp);
					break;
				case 4:
					//time = stoi(temp);
					break;
				case 5:
					//temp += line.at(i);
					temperature = stof(temp);	
					break;
				}
				temp.clear();
				dataColumn++;
			}
			else
			{
				temp += line.at(i);
			}

		}
		//cout << stationName << " " << year << " " << month << " " << day << " " << time << " " << temperature << "\n" << endl;
	}


};