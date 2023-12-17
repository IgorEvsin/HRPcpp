#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <functional>

// Функция считывает данные из CSV файла по URL
// На выходе выдает пару из вектора тикеров и вектора вектора значений таблицы
std::pair <std::vector<std::string>,std::vector<std::vector<double>>> readCSV(const std::string& filepath) {
    std::string tickers;
    std::vector<std::string> ticker_list;   // Список тикеров
    std::vector<std::string> dates;         // Список дат
    std::vector<std::vector<double>> data;  // Двумерный вектор для хранения цен из CSV без обозначения тикеров и дат

    std::ifstream file(filepath); // Открытие файла
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << filepath << std::endl;
        return std::pair <std::vector<std::string>,std::vector<std::vector<double>>> (ticker_list, data); // Возвращаем пустой вектор в случае ошибки открытия файла
    }

    std::string line;
    std::getline(file, tickers); // Запись первой строки в строку с тикерами
    
    std::stringstream tss(tickers);
    std::string ticker;
    std::getline(tss, ticker, ','); // Первый элемент пустой, пропускаем его
    while (std::getline(tss, ticker, ',')) { // Разбивка строки тикеров на ячейки по запятой
        ticker_list.push_back(ticker); // Добавление ячейки в вектор строки
        }

    while (std::getline(file, line)) { // Чтение каждой строки CSV
        std::vector<double> row; // Вектор для хранения значений строки

        std::stringstream ss(line);
        std::string date;
        std::getline(ss, date, ','); // Разбиение строки на дату по запятой
        dates.push_back(date); // Добавление даты в одномерный вектор, даты мы выводить не будем
        std::string cell;
        while (std::getline(ss, cell, ',')) { // Разбивка строки на ячейки по запятой
            double cell_d = std::stod(cell);
            row.push_back(cell_d); // Добавление ячейки в вектор строки
        }

        data.push_back(row); // Добавление строки в двумерный вектор
    }

    file.close(); // Закрытие файла
    
    std::pair <std::vector<std::string>,std::vector<std::vector<double>>> out = {ticker_list, data};

    return out; // Возврат пары из векторов названий тикеров и двумерного вектора данных из исходного CSV
}

// Класс методики портфельной оптимизации Hierarchical Risk Parity
// В конструктор передаем датасет со значениями цен
// Затем вызываем метод optimize, который выдает оптимальные веса
class HRP {
private:
    std::vector<std::vector<double>> data; // Приватный атрибут датасета
    
    // Метод рассчитывает доходности на основании датасета цен
    std::vector<std::vector<double>> calculateReturns(const std::vector<std::vector<double>>& prices) {
    std::vector<std::vector<double>> returns;

    for (std::size_t i = 0; i < prices.size(); ++i) {
        const std::vector<double>& currentRow = prices[i];
        std::vector<double> returnRow;

        if (i == 0) {
            returnRow.resize(currentRow.size(), 0.0);
        } else {
            const std::vector<double>& previousRow = prices[i - 1];

            for (std::size_t j = 0; j < currentRow.size(); ++j) {
                double priceReturn = (currentRow[j] - previousRow[j]) / previousRow[j];
                returnRow.push_back(priceReturn);
                }
            }

        returns.push_back(returnRow);
        }

    return returns;
    }

    // Считаем среднюю доходность активов
    double calculateMean(const std::vector<double>& data) {
    double sum = 0.0;
    for (const auto& value : data) {
        sum += value;
    }
    return sum / static_cast<double>(data.size());
    }

    // Считаем ковариацию между двумя векторами
    double calculateCovariance(const std::vector<double>& data1, const std::vector<double>& data2) {
        double mean1 = calculateMean(data1);
        double mean2 = calculateMean(data2);

        double covariance = 0.0;
        for (std::size_t i = 0; i < data1.size(); ++i) {
            covariance += (data1[i] - mean1) * (data2[i] - mean2);
        }

        return covariance / static_cast<double>(data1.size() - 1);
    }

    // Считаем стандартное отклонение вектора
    double calculateStandardDeviation(const std::vector<double>& data) {
        double mean = calculateMean(data);

        double variance = 0.0;
        for (const auto& value : data) {
            variance += std::pow(value - mean, 2);
        }

        return std::sqrt(variance / static_cast<double>(data.size() - 1));
    }

    // Считаем корреляционную матрицу
    std::vector<std::vector<double>> calculateCorrelationMatrix(const std::vector<std::vector<double>>& returns) {
        std::size_t numAssets = returns[0].size();
        std::vector<std::vector<double>> correlationMatrix(numAssets, std::vector<double>(numAssets, 0.0));

        for (std::size_t i = 0; i < numAssets; ++i) {
            for (std::size_t j = 0; j < numAssets; ++j) {
                std::vector<double> data1;
                std::vector<double> data2;

                for (const auto& row : returns) {
                    data1.push_back(row[i]);
                    data2.push_back(row[j]);
                }

                double covariance = calculateCovariance(data1, data2);
                double stdDev1 = calculateStandardDeviation(data1);
                double stdDev2 = calculateStandardDeviation(data2);

                correlationMatrix[i][j] = covariance / (stdDev1 * stdDev2);
            }
        }

        return correlationMatrix;
    }

    // Считаем ковариационную матрицу
    std::vector<std::vector<double>> calculateCovarianceMatrix(const std::vector<std::vector<double>>& data) {
        int n = data.size();
        int m = data[0].size();
        std::vector<std::vector<double>> covMatrix(m, std::vector<double>(m, 0.0));

        std::vector<double> mean(m, 0.0);
        for (const auto& row : data) {
            for (int j = 0; j < m; j++) {
                mean[j] += row[j];
            }
        }
        for (int j = 0; j < m; j++) {
            mean[j] /= n;
        }

        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                for (int i = 0; i < n; i++) {
                    covMatrix[j][k] += (data[i][j] - mean[j]) * (data[i][k] - mean[k]);
                }
                covMatrix[j][k] /= (n - 1);
            }
        }

        return covMatrix;
    }

    // Евклидово расстояние между двумя векторами
    double euclideanDistance(const std::vector<double>& col1, const std::vector<double>& col2) {
    if (col1.size() != col2.size()) {
        throw std::invalid_argument("Column sizes do not match");
    }

    double sum = 0.0;
    for (int i = 0; i < col1.size(); i++) {
        double diff = col1[i] - col2[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

    // Рассчитываем матрицу расстояний на основании матрицы корреляций и евклидовой метрики между колонками корреляций
    std::vector<std::vector<double>> calculateDistanceMatrix(const std::vector<std::vector<double>>& correlationMatrix) {
        double eps = 0.000000001;
        std::size_t numAssets = correlationMatrix.size();
        std::vector<std::vector<double>> distanceMatrix(numAssets, std::vector<double>(numAssets, 0.0));

        for (std::size_t i = 0; i < numAssets; ++i) {
            for (std::size_t j = 0; j < numAssets; ++j) {
                double correlation = correlationMatrix[i][j];
                double check = (1.0 - correlation) / 2.0;
                double distance = std::sqrt((1.0 - correlation) / 2.0 + eps);
                distanceMatrix[i][j] = distance;
            }
        }

    std::vector<std::vector<double>> EuclidianMatrix(numAssets, std::vector<double>(numAssets, 0.0));

    for (std::size_t i = 0; i < numAssets; i++) {
        for (std::size_t j = 0; j < numAssets; j++) {
            double distance = euclideanDistance(distanceMatrix[i], distanceMatrix[j]);
            // std:: cout << distance << " " << i << " " << j << std::endl;
            EuclidianMatrix[i][j] = distance;
        }
    }

        return EuclidianMatrix;
    }

    // Агломеративная кластеризация, аналог scipy.cluster.hierarchy.linkage с single linkage связью
    std::vector<std::vector<double>> agglomerativeClustering(std::vector<std::vector<double>> matrix) {
        int n = matrix.size();
        // Создание копии матрицы расстояний для следующей итерации
        std::vector<std::vector<double>> new_matrix(matrix);
    
        // Инициализация вектора cluster_size
        std::vector<int> cluster_size(n * 2, 1);
    
        std::vector<std::vector<double>> result;
    
        for (int step = 0; step < n - 1; ++step) {
            int s = n + step;
            
            // Поиск минимального значения в матрице, не находящегося на главной диагонали
            double min_value = std::numeric_limits<double>::infinity();
            int min_i, min_j;
            
            for (int i = 0; i < new_matrix.size(); ++i) {
                for (int j = 0; j < new_matrix[i].size(); ++j) {
                    if (i != j && new_matrix[i][j] < min_value) {
                        min_value = new_matrix[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }
            
            // Создание нового вектора минимумов
            std::vector<double> min_vector;
            
            for (int k = 0; k < new_matrix[min_i].size(); ++k) {
                min_vector.push_back(std::min(new_matrix[min_i][k], new_matrix[min_j][k]));
            }
            
            min_vector.push_back(0.0);
            
            // Добавление новой строки и столбца в матрицу расстояний
            new_matrix.push_back(min_vector);
            
            for (int i = 0; i < new_matrix.size()-1; ++i) {
                new_matrix[i].push_back(min_vector[i]);
            }
            
            // Обновление значения cluster_size
            cluster_size[s] = cluster_size[min_i] + cluster_size[min_j];
            
            // Замена значений в строках и столбцах i, j на infinity
            for (int i = 0; i < new_matrix.size(); ++i) {
                new_matrix[min_i][i] = std::numeric_limits<double>::infinity();
                new_matrix[min_j][i] = std::numeric_limits<double>::infinity();
                new_matrix[i][min_i] = std::numeric_limits<double>::infinity();
                new_matrix[i][min_j] = std::numeric_limits<double>::infinity();
            }
            
            // Добавление результатов текущего шага в результаты алгоритма
            result.push_back({(double) min_i, (double) min_j, (double) min_value, (double) cluster_size[s] });
        }
    
        return result;
    }


    // На основании полученной иерархической кластеризации создаем индексы для последующей квазидиагонализации ковариационной матрицы
    std::vector<int> getQuasiDiag(std::vector<std::vector<double>>& link) {
        for (auto& v : link) {
            for (auto& i : v) {
                i = static_cast<int>(i);
            }
        }

        int numItems = static_cast<int>(link.back()[3]);
        int n = numItems; // Число изначальных активов

        std::vector<int> sortIx(n, -1);

        // Сортируем кластеры на основании расстояния
        for (int i = 0; i < link.size(); i++) {
            int item1 = static_cast<int>(link[i][0]);
            int item2 = static_cast<int>(link[i][1]);
            
            if (sortIx[item1] == -1) {
                sortIx[item1] = i;
            }
            
            if (sortIx[item2] == -1) {
                sortIx[item2] = i;
            }
        }
        
        // Заменяем кластеры с их индексами и поправляем индексы оставшихся элементов
        int clusterIndex = link.size();
        for (int i = 0; i < sortIx.size(); i++) {
            if (sortIx[i] == -1) {
                sortIx[i] = clusterIndex;
                clusterIndex++;
            }
        }

        return sortIx;
    }


    std::vector<double> getIVP(const std::vector<std::vector<double>>& cov) {
        // Считаем inverse-variance портфель
        int n = cov.size();
        std::vector<double> ivp(n);
        double totalSum = 0.0;

        for (int i = 0; i < n; i++) {
            ivp[i] = 1.0 / cov[i][i];
            totalSum += ivp[i];
        }

        for (int i = 0; i < n; i++) {
            ivp[i] /= totalSum;
        }

        return ivp;
    }

    double getClusterVar(const std::vector<std::vector<double>>& cov, const std::vector<int>& cItems) {
        // Считаем дисперсию кластера
        int n = cItems.size();
        std::vector<std::vector<double>> covSlice(n, std::vector<double>(n, 0.0));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                covSlice[i][j] = cov[cItems[i]][cItems[j]];
            }
        }

        std::vector<double> w = getIVP(covSlice);
        double cVar = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cVar += w[i] * covSlice[i][j] * w[j];
            }
        }

        return cVar;
    }

    std::vector<double> getRecBipart(const std::vector<std::vector<double>>& cov, const std::vector<int>& sortIx) {
        // Считаем веса согласно HRP методике
        int n = sortIx.size();
        std::vector<double> w(n, 1.0);
        std::vector<std::vector<int>> cItems = {sortIx};

        while (!cItems.empty()) {
            std::vector<std::vector<int>> newCItems;

            for (const auto& item : cItems) {
                if (item.size() > 1) {
                    int mid = item.size() / 2;
                    std::vector<int> cItems0(item.begin(), item.begin() + mid);
                    std::vector<int> cItems1(item.begin() + mid, item.end());
                    newCItems.push_back(cItems0);
                    newCItems.push_back(cItems1);

                    double cVar0 = getClusterVar(cov, cItems0);
                    double cVar1 = getClusterVar(cov, cItems1);
                    double alpha = 1.0 - cVar0 / (cVar0 + cVar1);

                    for (int i = 0; i < cItems0.size(); i++) {
                        w[cItems0[i]] *= alpha;
                    }

                    for (int i = 0; i < cItems1.size(); i++) {
                        w[cItems1[i]] *= (1.0 - alpha);
                    }
                }
            }

            cItems = newCItems;
        }

        std::vector<double> temp;
        double sum_w = 0;
        for (int i=0; i < w.size(); i++) {
            if (w[i] == 1) temp.push_back(0);
            else {temp.push_back(w[i]); sum_w+=w[i];};
        }

        for (int i=0; i < temp.size(); i++) {
            temp[i] = temp[i] / sum_w;
        }

        return temp;
    };

public:
    HRP(std::vector<std::vector<double>> data) : data(data) {}

    std::vector<double> optimize(){

    std::vector<std::vector<double>> returns = calculateReturns(data);
    std::vector<std::vector<double>> corrs = calculateCorrelationMatrix(returns);
    std::vector<std::vector<double>> covs = calculateCovarianceMatrix(returns);
    std::vector<std::vector<double>> dist = calculateDistanceMatrix(corrs);
    std::vector<std::vector<double>> results = agglomerativeClustering(dist);
    std::vector<int> quasiDiagIds = getQuasiDiag(results);
    std::vector<double> weights = getRecBipart(covs, quasiDiagIds);
    return weights;
    }
};

int main() {
    std::string filepath = "sp500_adj_close.csv";
    std::pair <std::vector<std::string>,std::vector<std::vector<double>>> read_pair = readCSV(filepath);
    
    std::vector<std::string> tickers = read_pair.first;
    std::vector<std::vector<double>> data = read_pair.second;
    
    HRP hrp(data);
    std::vector<double> weights = hrp.optimize();

    std::cout << "Hierarchical Risk Parity Weights: " << std:: endl;
    for (int i = 0; i < tickers.size()-1; i++) {
        std::cout << tickers[i] << ": " << weights[i] << std::endl;
    }

    return 0;
}
