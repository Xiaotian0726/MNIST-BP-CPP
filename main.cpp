#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

/*
    2020-07-09 1:20
    studying git
    on brance dev

    1

    after stash
*/

vector< vector<double> > train_images;        // 训练集图像
vector<double> train_labels;                  // 训练集标签
vector< vector<double> > test_images;         // 测试集图像
vector<double> test_labels;                   // 测试集标签

const double learning_rate = 0.08;      // 学习率
const int epoch = 15;                    // 学习轮次
const int nh = 30;                      // 隐藏单元数量（单隐藏层）
double w1[784][nh];                     // 输入层到隐藏层的权重矩阵
double w2[nh][10];                      // 隐藏层到输出层的权重矩阵
double bias1[nh];                       // 隐藏层的偏置
double bias2[10];                       // 输出层的偏置

int e;

void test();

int reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_train_images() {                 // 载入训练集图像
	ifstream file("train-images-idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;
		
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));
		
		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);
		
		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;          // 像素值归一化处理
				}
			}
			train_images.push_back(this_image);
		}
		printf("%d, train images success\n", train_images.size());
	}
}
 
void read_train_labels() {                   // 载入训练集标签
	ifstream file;
	file.open("train-labels-idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));

		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			train_labels.push_back((double)label);
		}
		printf("%d, train labels success\n", train_labels.size());
	}
}

void read_test_images() {                  // 载入测试集图像
	ifstream file("t10k-images-idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int row = 0;
		int col = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&row, sizeof(row));
		file.read((char*)&col, sizeof(col));
		
		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);
		row = reverse_int(row);
		col = reverse_int(col);

		for (int i = 0; i < number_of_images; i++) {
			vector<double> this_image;
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(pixel));
					this_image.push_back(pixel);
					this_image[r * 28 + c] /= 255;          // 像素值归一化处理
				}
			}
			test_images.push_back(this_image);
		}
		printf("%d, test images success\n", test_images.size());
	}
}

void read_test_labels() {                       // 载入测试集标签
	ifstream file;
	file.open("t10k-labels-idx1-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		
		magic_number = reverse_int(magic_number);
		number_of_images = reverse_int(number_of_images);

		for (int i = 0; i < number_of_images; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			test_labels.push_back((double)label);
		}
		printf("%d, test labels success\n", test_labels.size());
	}
}

void init_parameters() {                      // 为权重矩阵和偏置向量随机赋初值
	for (int i = 0; i < 784; i++) {
		for (int j = 0; j < nh; j++) w1[i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}
	
	for (int i = 0; i < nh; i++) {
		for (int j = 0; j < 10; j++) w2[i][j] = rand() / (10 * (double)RAND_MAX) - 0.05;
	}
	
	for (int i = 0; i < nh; i++) bias1[i] = rand() / (10 * (double)RAND_MAX) - 0.1;
	for (int i = 0; i < 10; i++) bias2[i] = rand() / (10 * (double)RAND_MAX) - 0.1;
}

// 激活函数sigmoid，可以把x映射到0~1之间
// s(x) = 1 / (1 + e^(-x))
// s'(x) = s(x) * (1 - s(x))
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

// 通过图像的输入得到隐藏层的输出
vector<double> get_hidden_out(vector<double>& image) {
	vector<double> hidden_out(nh);
	for (int i = 0; i < nh; i++) {          // 对于每一个隐藏单元
		double sum = 0.0;
		for (int j = 0; j < 784; j++) {
			sum += image[j] * w1[j][i];
		}
		hidden_out[i] = sigmoid(sum + bias1[i]);
	}
	return hidden_out;
}

// 通过隐藏层的输出得到网络最终的输出
vector<double> get_z(vector<double>& hidden_out) {
	vector<double> z(10);
	for (int i = 0; i < 10; i++) {        // 对于每一个输出单元
		double sum = 0.0;
		for (int j = 0; j < nh; j++) {
			sum += hidden_out[j] * w2[j][i];
		}
		z[i] = sigmoid(sum + bias2[i]);
	}
	return z;
}

// 计算损失函数（1/2均方误差）
double get_loss(vector<double>& z, double label) {
	double loss = 0;
	int true_label = (int)label;
	for (int i = 0; i < 10; i++) {
		if (i != true_label) loss += z[i] * z[i];
		else loss += (1 - z[i]) * (1 - z[i]);
	}
	return loss / 2;
}

void train_and_test() {
	for (e = 1; e <= epoch; e++) {
		for (int im = 0; im < train_images.size(); im++) {
			double grad_w1[784][nh];
			double grad_w2[nh][10];
			double grad_bias1[nh];
			double grad_bias2[10];
			
			vector<double> hidden_out = get_hidden_out(train_images[im]);
			vector<double> z = get_z(hidden_out);
			double loss = get_loss(z, train_labels[im]);
			int true_label = (int)train_labels[im];
			//printf("epoch = %d, image = %d, loss = %f\n", e, im + 1, loss);
			
			// -------------------------------------计算梯度----------------------------------
			
			for (int i = 0; i < 784; i++) {
				for (int j = 0; j < nh; j++) {
					double sum = 0.0;
					for (int k = 0; k < 10; k++) {
						double labelk = (k == true_label) ? 1.0 : 0.0;
						sum += (z[k] - labelk) * z[k] * (1 - z[k]) * w2[j][k] * hidden_out[j] * (1 - hidden_out[j]) * train_images[im][i];
					}
					grad_w1[i][j] = sum;
				}
			}
			
			for (int j = 0; j < nh; j++) {
				for (int k = 0; k < 10; k++) {
					double labelk = (k == true_label) ? 1.0 : 0.0;
					grad_w2[j][k] = (z[k] - labelk) * z[k] * (1 - z[k]) * hidden_out[j];
				}
			}
			
			for (int j = 0; j < nh; j++) {
				double sum = 0.0;
				for (int k = 0; k < 10; k++) {
					double labelk = (k == true_label) ? 1.0 : 0.0;
					sum += (z[k] - labelk) * z[k] * (1 - z[k]) * w2[j][k] * hidden_out[j] * (1 - hidden_out[j]);
				}
				grad_bias1[j] = sum;
			}
			
			for (int k = 0; k < 10; k++) {
				double labelk = (k == true_label) ? 1.0 : 0.0;
				grad_bias2[k] = (z[k] - labelk) * z[k] * (1 - z[k]);
			}
			
			// -------------------------------------更新权与偏置----------------------------------
			
			for (int i = 0; i < 784; i++) {
				for (int j = 0; j < nh; j++) {
					w1[i][j] -= learning_rate * grad_w1[i][j];
				}
			}
			
			for (int i = 0; i < nh; i++) {
				for (int j = 0; j < 10; j++) {
					w2[i][j] -= learning_rate * grad_w2[i][j];
				}
			}
			
			for (int i = 0; i < nh; i++) {
				bias1[i] -= learning_rate * grad_bias1[i];
			}
			
			for (int i = 0; i < 10; i++) {
				bias2[i] -= learning_rate * grad_bias2[i];
			}
		}
		test();
	}
}

void test() {
	int cnt = 0;
	for (int i = 0; i < test_images.size(); i++) {
		int true_label = (int)test_labels[i];
		vector<double> hidden_out = get_hidden_out(test_images[i]);
		vector<double> z = get_z(hidden_out);
		int recognize = -1;
		double max = 0;
		for (int i = 0; i < 10; i++) {
			if (z[i] > max) {
				max = z[i];
				recognize = i;
			}
		}
		//printf("test image %d, predict = %d, true = %d\n", i + 1, recognize, true_label);
		if (recognize == true_label) cnt++;
	}
	printf("epoch = %d, precision = %f\n", e, (double)cnt / test_images.size());
}

int main() {
	read_train_images();                // 载入训练集图像
	read_train_labels();                // 载入训练集标签
	read_test_images();                 // 载入测试集图像
	read_test_labels();                 // 载入测试集标签
	init_parameters();                  // 为权重矩阵和偏置向量随机赋初值
	train_and_test();	
	return 0;
}
