#ifndef JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_
#define JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_

#define CV_MAJOR_VERSION CV_MAJOR_VERSION 
#define CV_MINOR_VERSION CV_MINOR_VERSION
#define CV_SUBMINOR_VERSION CV_SUBMINOR_VERSION

#include <map>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "jubatus/core/fv_converter/binary_feature.hpp"
#include "jubatus/core/fv_converter/exception.hpp"

#include <iostream>


namespace jubatus{
namespace plugin{
namespace fv_converter{

using core::fv_converter::converter_exception;

class image_feature:public jubatus::core::fv_converter::binary_feature{
public:
	virtual ~image_feature(){}
	image_feature(
		const std::string& algorithm = "RGB",
		const bool resize = false,
		float x_size = 50.0,
		float y_size = 50.0
		);
	void add_feature(
		const std::string& key,
		const std::string& value,
		std::vector<std::pair<std::string, float> >& ret_fv) const; 

	void dense_sampler(
		const cv::Mat mat,
		const int step,
		std::vector<cv::KeyPoint>& keypoint) const;

private:
	std::string algorithm_;
	bool resize_;
	float x_size_;
	float y_size_;
};


}  // namespace fv_converter
}  // namespace plugin
}  // namespace jubatus


extern "C" {
jubatus::plugin::fv_converter::image_feature* create(
		const std::map<std::string, std::string>& params);
std::string version();
} //extern "C"

#endif //JUBATUS_PLUGIN_FV_CONVERTER_IMAGE_FEATURE_HPP_
