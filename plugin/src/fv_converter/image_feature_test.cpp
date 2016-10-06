
// #include <map>
// #include <string>
// #include <utility>
// #include <vector>
 #include "jubatus/util/concurrent/thread.h"
// #include "jubatus/util/lang/bind.h"
// #include "jubatus/util/lang/scoped_ptr.h"
#include "jubatus/core/fv_converter/exception.hpp"
#include <gtest/gtest.h>
#include <iterator>
#include <fstream>
#include <sstream>
#include "image_feature.hpp"


namespace jubatus {
namespace plugin {
namespace fv_converter {

TEST(image_feature, trivial){
	jubatus::plugin::fv_converter::image_feature im;
	cv::Mat img = cv::imread("test_input/jubatus.jpg");
	int correct = img.cols * img.rows * 3;
	//read a file
	std::ifstream ifs("test_input/jubatus.jpg");
	if(!ifs){
		std::cerr << "cannot open : test_input/jubatus.jpg"  << std::endl;
		exit(1);
	}
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	
	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature("jubatus", buffer.str(), ret_fv);

	ASSERT_EQ(correct, ret_fv.size());
}

TEST(image_feature, ORBdefault){
	jubatus::plugin::fv_converter::image_feature im("ORB");
	cv::Mat img = cv::imread("test_input/jubatus.jpg");
	int correct = (img.cols-24) * (img.rows-24) * 32;
	//read a file
	std::ifstream ifs("test_input/jubatus.jpg");
	if(!ifs){
		std::cerr << "cannot open : test_input/jubatus.jpg"  << std::endl;
		exit(1);
	}
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	
	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature("jubatus", buffer.str(), ret_fv);

	ASSERT_EQ(correct, ret_fv.size());
}


TEST(image_feature, RGB_resize){
	jubatus::plugin::fv_converter::image_feature im("RGB",true);
	cv::Mat img = cv::imread("test_input/jubatus.jpg");
	int correct = 50 * 50 * 3;
	//read a file
	std::ifstream ifs("test_input/jubatus.jpg");
	if(!ifs){
		std::cerr << "cannot open : test_input/jubatus.jpg"  << std::endl;
		exit(1);
	}
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	
	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature("jubatus", buffer.str(), ret_fv);

	ASSERT_EQ(correct, ret_fv.size());
}

TEST(image_feature, ORB_resize){
	jubatus::plugin::fv_converter::image_feature im("ORB",true);
	cv::Mat img = cv::imread("test_input/jubatus.jpg");
	int correct = (50-24) * (50-24) * 32;
	//read a file
	std::ifstream ifs("test_input/jubatus.jpg");
	if(!ifs){
		std::cerr << "cannot open : test_input/jubatus.jpg"  << std::endl;
		exit(1);
	}
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	
	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature("jubatus", buffer.str(), ret_fv);

	ASSERT_EQ(correct, ret_fv.size());
}

TEST(image_feature, size_designate){
	jubatus::plugin::fv_converter::image_feature im("ORB",true,30,40);
	cv::Mat img = cv::imread("test_input/jubatus.jpg");
	int correct = (30-24) * (40-24) * 32;
	//read a file
	std::ifstream ifs("test_input/jubatus.jpg");
	if(!ifs){
		std::cerr << "cannot open : test_input/jubatus.jpg"  << std::endl;
		exit(1);
	}
	std::stringstream buffer;
	buffer << ifs.rdbuf();
	
	std::vector<std::pair<std::string, float> > ret_fv;
	im.add_feature("jubatus", buffer.str(), ret_fv);

	ASSERT_EQ(correct, ret_fv.size());
}

}  // namespace fv_converter
}  // namespace plugin
}  // namespace jubatus
