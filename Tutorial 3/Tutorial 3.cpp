#define CL_USE_DEPRECATED_OPENCL_1_2_APIS //JB
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
#include "WeatherDataElement.cpp"
using namespace std;
//Read in Data
int LoadFile(char* directory, vector<WeatherDataElement*> &weatherData)
{
	ifstream file(directory, ios::in);
	int numOfElements = 0;
	string line;	
	
	if (file)
	{
		std::cout << "File Opened!" << std::endl;
		while (getline(file, line))
		{
			WeatherDataElement* temp = new WeatherDataElement();
			temp->Instantiate(line);
			weatherData.push_back(temp);
			//cout << line << endl;
			numOfElements++;
		}
	}
	else cout << "File Not Opened!" << endl;
	file.close();
	return numOfElements;
}

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//instanciate the weather data
	std::vector<WeatherDataElement*> WeatherData;

	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//Trys until an exxection/ error//JB
	try {
		//use ifstream to read in data
		//Read in data using ifStream
		//populate the weatherData vector with each column of data from the file
		//It reutnrs the number of elements, which is used for padding later.
		int dataSize = LoadFile("../../temp_lincolnshire.txt", WeatherData);
		
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		int value = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		cout << "devuce max group size: " << value << endl;


		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue of commands for the device
		cl::CommandQueue queue(context);	

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "Weather_Kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		typedef float myfloat;
		typedef int myint;

		//Part 4 - memory allocation
		std::vector<myfloat> WeatherVector(dataSize, 0);
		//populate memory vectors with temp data, from the list of all data
		for (int i = 0; i < dataSize; i++)
		{
			WeatherVector[i] = WeatherData[i]->temperature;
		}
		//pad the data so it is a multipul of the local_size, this makes it more efficiant
		size_t local_size = 32; //32 

		size_t padding_size = WeatherVector.size() % local_size;

		//if the input vector is not a multiple of the local_size then add 99's to the end, to pad the data. 99s are used becuase they are never placed in any bin. 
		if (padding_size) {
			//create a placeholder verctor to contain the extra values, and then append them to the end of the weather vector
			std::vector<float> WeatherVector_ext(local_size - padding_size,99);			
			WeatherVector.insert(WeatherVector.end(), WeatherVector_ext.begin(), WeatherVector_ext.end());
		}
		//instanciate values used for buffer creation and keeping tabs on work group sizes and number of workgroups
		size_t input_elements = WeatherVector.size();
		size_t input_size = input_elements * sizeof(myfloat);
		size_t nr_groups = input_elements / local_size;
		size_t output_size = nr_groups * sizeof(myfloat);
		size_t output_size_Hist = (nr_groups *4) * sizeof(myint);
		
		//vectors used in calculatoins
		std::vector<myfloat> B(output_size);
		std::vector<myfloat> C(output_size);
		std::vector<myfloat> D(input_elements);
		std::vector<myint> E(output_size_Hist);

		size_t fmean_output_size = C.size() * sizeof(myfloat);
		//___________________________________DEVICE_BUFFERS_______________________________________

		// create buffers, and fill them. some will be larger then others, due to having to contain histogram data, 4 bins, per work group.
		cl::Buffer buffer_Weather(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_Output(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Output_Hist(context, CL_MEM_READ_WRITE, output_size_Hist);
		cl::Buffer buffer_AverageOutput(context, CL_MEM_READ_WRITE, fmean_output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size_Hist);

		//Fill them, or at teast designate the size.
		queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, input_size, &WeatherVector[0]);
		queue.enqueueFillBuffer(buffer_Output, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_Output_Hist, 0, 0, output_size_Hist);
		queue.enqueueFillBuffer(buffer_AverageOutput, 0, 0, fmean_output_size);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, input_size, &D[0]);
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size_Hist, &E[0]);
				
		//________________________________FLOAT_CALCULATIONS_______________________________________
		
		//		  ____________________________REDUCE_MAX_________________________________
		int temp_nr_groups;
		//repeate the reduction until its no longer wother reducing.
		while (nr_groups>2)
		{
			cout << "nr_groups: " << nr_groups << endl;
			temp_nr_groups = nr_groups;		
			//reduction using local memory
			cl::Kernel kernel_fmin = cl::Kernel(program, "reduce_max");
			kernel_fmin.setArg(0, buffer_Weather);
			kernel_fmin.setArg(1, buffer_Output);
			kernel_fmin.setArg(2, cl::Local(local_size * sizeof(myfloat)));
			queue.enqueueNDRangeKernel(kernel_fmin, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, output_size, &B[0]);

			C = B;

			size_t padding_size = B.size() % local_size;
			//fil the input buffer with the output. and repeate the kernel.
			queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, B.size(), &B[0]);
			nr_groups = nr_groups / local_size;	//calculation to get the relavent workgroups next turn.

		}
		//cout << temp_nr_groups << endl;
		for (int i = 0; i < temp_nr_groups; i++)	// process the vector up to the element number equivilent to the number of workgroups of the last reduction.
		{
			if (C[i] > C[0])
				C[0] = C[i];
		}

		cout << "Maximum floating : " << C[0] << endl;

		//		  ____________________________REDUCE_MIN_________________________________

		//..A repeate of the Max reduction . the only difference being the comarasion in the kernel and at the end of the while loop.

		queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, input_size, &WeatherVector[0]);
		nr_groups = input_elements / local_size;
		while (nr_groups>2)
		{
			cout << "nr_groups: " << nr_groups << endl;
			temp_nr_groups = nr_groups;
			
			cl::Kernel kernel_fmin = cl::Kernel(program, "reduce_min");
			kernel_fmin.setArg(0, buffer_Weather);
			kernel_fmin.setArg(1, buffer_Output);
			kernel_fmin.setArg(2, cl::Local(local_size * sizeof(myfloat)));
			queue.enqueueNDRangeKernel(kernel_fmin, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
			queue.enqueueReadBuffer(buffer_Output, CL_TRUE, 0, output_size, &B[0]);
	
			C = B;	//hang onto the ouput data before it is reset.

			size_t padding_size = B.size() % local_size;

			queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, B.size(), &B[0]);

			nr_groups = nr_groups / local_size;

		}
		for (int i = 0; i < temp_nr_groups; i++)	// process the vector up to the element number equivilent to the number of workgroups of the last reduction.
		{
			if (C[i] < C[0])
				C[0] = C[i];
		}

		cout << "Minimum floating : " << C[0] << endl;
		//		  ____________________________REDUCE_MEAN_________________________________
		//A more limited reduction this time. only runs the kernel once. 
		queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, input_size, &WeatherVector[0]);
		nr_groups = input_elements / local_size;
		float sum = 0.0f;
		
		// float mean using reduction to return array of work group sums
		// fast on current data set but additional kernals could be used to further reduce the data
		// work group sums and added and then devied by the datasize to get the mean
		cl::Kernel kernel_fmean = cl::Kernel(program, "reduce_sum");
		kernel_fmean.setArg(0, buffer_Weather);
		kernel_fmean.setArg(1, buffer_AverageOutput);
		kernel_fmean.setArg(2, cl::Local(local_size * sizeof(myfloat)));//local memory size
		queue.enqueueNDRangeKernel(kernel_fmean, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
		queue.enqueueReadBuffer(buffer_AverageOutput, CL_TRUE, 0, fmean_output_size, &C[0]);
		float fsum = 0.0f;
		for (int i = 0; i <= nr_groups; i++) //sums the reduction outputs,
		{
			fsum += C[i]; 
		}
		float fmean = fsum / (dataSize);	//get the average from the sum
		cout << "Mean floating : " << fmean << endl;	
		
		//		  ____________________________SORT_&_HISTOGRAM_________________________________

		//I inialy tried to sort and then search for the histogram. however On the large data set, the device runs out of memory. 
		//A solution would be to split the large data set into smaller parts. then process them inipendently. 
		//Sort works on the smaller dataset.

		//cl::Kernel kernel_Sort = cl::Kernel(program, "selection_sort_local_float");
		//kernel_Sort.setArg(0, buffer_Weather);
		//kernel_Sort.setArg(1, buffer_D);
		//kernel_Sort.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		//queue.enqueueNDRangeKernel(kernel_Sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
		//queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, input_size, &D[0]);
		///*for each (float f in D)
		//{
		//	cout << f << " ";
		//}
		//cout << endl;*/
		
		queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, WeatherVector.size(), &WeatherVector[0]);	//refresh the data set, just a precausion. 
		
		cout << endl;

		cl::Kernel kernel_Search = cl::Kernel(program, "selection_search_local_float");
		kernel_Search.setArg(0, buffer_Weather);	//filter in the data
		kernel_Search.setArg(1, buffer_E);			//place the outputed data into a larger buffer, 4 times the size of the input buffer.
		kernel_Search.setArg(2, cl::Local(local_size * sizeof(myfloat)));
		kernel_Search.setArg(3, cl::Local(4 * sizeof(myint)));
		queue.enqueueNDRangeKernel(kernel_Search, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_size_Hist, &E[0]);
		//cout<<
		/*for each (int i in E)
		{
			cout << i << " ";
			
		}
		cout << endl*/;

		for (int i = 0; i <= nr_groups; i++)	//concatinate the 4 outputs from all workgroups into 4 bins. 
		{
			E[0] += E[i * 4];
			E[1] += E[(i * 4) + 1];		//access data every 4th bin.
			E[2] += E[(i * 4) + 2];
			E[3] += E[(i * 4) + 3];
			
		}
		cout << "E0: " << E[0] << endl;	//output.
		cout << "E1: " << E[1] << endl;
		cout << "E2: " << E[2] << endl;
		cout << "E3: " << E[3] << endl;
		//cout << "Data size: " << WeatherData.size() << endl;

		//std::vector<myint> intWeatherVector(dataSize, 0);
		////populate memory vectors
		//for (int i = 0; i < dataSize; i++)
		//{
		//	intWeatherVector[i] = WeatherVector[i];
		//}

		//queue.enqueueWriteBuffer(buffer_Weather, CL_TRUE, 0, input_size, &intWeatherVector[0]);

		//cout << "Histogram." << endl;
		//kernel_fmean = cl::Kernel(program, "hist_atomic");
		//kernel_fmean.setArg(0, buffer_Weather);
		//kernel_fmean.setArg(1, buffer_Output);
		//kernel_fmean.setArg(2, cl::Local(local_size * sizeof(myint)));//local memory size
		//queue.enqueueNDRangeKernel(kernel_fmean, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, NULL);
		//queue.enqueueReadBuffer(buffer_AverageOutput, CL_TRUE, 0, fmean_output_size, &C[0]);

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		//system("clear");
	}

	
	return 0;
}
