#include "image_feature.hpp"
#include <iterator>
#include <fstream>
#include <sstream>

int main(int argc, char** argv){
	//check command line args.
	if(argc != 2){
		std::cerr << "set target filename" << std::endl;
		return 1;
	}
	//read a file
	std::ifstream ifs(argv[1]);
	if(!ifs){
		std::cerr << "cannot open : " << argv[1] << std::endl;
		return -1;
	}

	std::stringstream buffer;
	buffer << ifs.rdbuf();

	jubatus::plugin::fv_converter::image_feature im(1.0,"RGB");

	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature(argv[1], buffer.str(), ret_fv);
	for (int i=0;i<ret_fv.size();++i){
		std::cout << ret_fv[i].first
				  << ":"
				  << ret_fv[i].second << std::endl;
	}
	std::cout << ret_fv.size() << std::endl;
	std::cout << ret_fv[1].first << std::endl;
	std::cout << ret_fv[0].second << std::endl;

}
