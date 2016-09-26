#include "image_feature.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "jubatus/util/data/string/utility.h"
#include "jubatus/core/fv_converter/exception.hpp"
#include "jubatus/core/fv_converter/util.hpp"

namespace jubatus {
namespace plugin {
namespace fv_converter {

image_feature::image_feature(
	float x_size,
	float y_size,
	const std::string& algorithm)
	: x_size_(x_size),
	  y_size_(y_size),
	  algorithm_(algorithm){
	//some exceptions
}


void image_feature::dense_sampler(
	const cv::Mat mat,
	const int step,
	std::vector<cv::KeyPoint>& kp_vec) const {
	for (int y=step; y<mat.rows-step; y+=step){
		for (int x=step; x<mat.cols-step; x+=step){
				kp_vec.push_back(cv::KeyPoint(float(x),float(y),float(step)));
		}
	}
}


void image_feature::add_feature(
	const std::string& key,
	const std::string& value,
	std::vector<std::pair<std::string,float> >& ret_fv) const{
	std::vector<unsigned char> buf(value.begin(), value.end());
	cv::Mat mat_orig = cv::imdecode(cv::Mat(buf), CV_LOAD_IMAGE_COLOR);
	
	// mat resize and gray scale for DENSE sampling
	cv::Mat mat_resize;
	cv::Mat mat_gray;
	float m_x = x_size_ / mat_orig.cols;
	float m_y = y_size_ / mat_orig.rows;
	cv::resize(mat_orig, mat_resize, cv::Size(), m_x , m_y);
	cv::cvtColor(mat_resize, mat_gray, CV_BGR2GRAY); 
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> kp_vec;

	// feature extractors
	if (algorithm_ == "RGB"){
		for (int y = 0; y < mat_resize.rows; ++y) {
			for (int x = 0; x < mat_resize.cols; ++x) {
				const cv::Vec3b& vec = mat_resize.at<cv::Vec3b>(y, x);
				for (int c = 0; c < 3; ++c) {
					std::ostringstream oss;
					oss << key << '-' << x << '-' << y << '-' << c;
					float val = static_cast<float>(vec[c]) / 255.0;
					ret_fv.push_back(std::make_pair(oss.str(), val));
				}
			}
		}
	}
	else if (algorithm_ == "ORB"){
		dense_sampler(mat_gray,1,kp_vec);
		cv::Ptr<cv::Feature2D> extractor = cv::ORB::create(500,1.2f,8,12,0,2,0,31);
	    extractor->compute(mat_gray, kp_vec, descriptors);
	}else if (algorithm_ == "BRISK"){
		dense_sampler(mat_gray,1,kp_vec);
		cv::Ptr<cv::Feature2D> extractor = cv::BRISK::create();
	    extractor->compute(mat_gray, kp_vec, descriptors);
	}else if (algorithm_ == "AKAZE"){
		cv::Ptr<cv::Feature2D> akaze = cv::AKAZE::create();;
		akaze->detect(mat_gray,kp_vec);
		akaze->compute(mat_gray,kp_vec,descriptors);
	}else{
		throw JUBATUS_EXCEPTION(
	        converter_exception("input algorithm among these ... RGB, ORB, BRISK, AKAZE"));
	}
    // *logs
	// std::cout << "keypoints of img: " << kp_vec.size() << std::endl;
    // std::cout << "x_ori "<< mat_orig.cols <<",y_ori " << mat_orig.rows <<std::endl;
	// std::cout << "m_x "<< m_x <<",m_y " << m_y <<std::endl;
    // std::cout << "x_res "<< mat_resize.cols <<",y_res " << mat_resize.rows <<std::endl;
    // std::cout << "descriptors of img: " << descriptors.size() << std::endl;
    for (int i = 0; i < descriptors.rows; ++i){
    	for (int j = 0; j < descriptors.cols; ++j){
			std::ostringstream oss;
			int p = descriptors.at<uchar>(i,j);
			oss << key << "-"<< i << "-" << j << "-" << p ;
			ret_fv.push_back(std::make_pair(oss.str(),p));
		}    	
	}

}
}  // namespace fv_converter
}  // namespace plugin
}  // namespace jubatus

extern "C" {
jubatus::plugin::fv_converter::image_feature* create(
		const std::map<std::string, std::string>& params) {
	using jubatus::util::lang::lexical_cast;
	using jubatus::core::fv_converter::get_with_default;
	
	float x_size = lexical_cast<float>(get_with_default(params,"x_size","50.0"));
	float y_size = lexical_cast<float>(get_with_default(params,"y_size","50.0"));
	std::string algorithm = get_with_default(params,"algorithm","RGB");
	
	return new jubatus::plugin::fv_converter::image_feature(
		x_size, y_size, algorithm);
}
std::string version(){
	return "1.0";
}

} //extern "C"
