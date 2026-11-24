"""
==============================================================
  app_constants.py – Danh mục cây trồng và thông số kỹ thuật
  Dự án: Crop Recommendation System
==============================================================
"""

# Từ điển thông tin nông học cho 22 loại cây trồng
CROP_INFO = {
    "rice": {
        "vi": "Lúa",
        "scientific": "Oryza sativa",
        "category": "Cây lương thực",
        "soil_type": "Đất phù sa, đất sét pha thịt, giữ nước tốt",
        "climate_requirement": "Lượng mưa > 200 mm, độ ẩm > 80%, nhiệt độ 20 - 27°C, pH trung tính (6.0 - 7.0)."
    },
    "maize": {
        "vi": "Ngô",
        "scientific": "Zea mays",
        "category": "Cây lương thực",
        "soil_type": "Đất thịt nhẹ, đất cát pha, tơi xốp, thoát nước nhanh",
        "climate_requirement": "Nhiệt độ 18 - 27°C, lượng mưa vừa phải (60 - 100 mm), độ ẩm 55 - 75%."
    },
    "chickpea": {
        "vi": "Đậu gà",
        "scientific": "Cicer arietinum",
        "category": "Cây họ đậu",
        "soil_type": "Đất cát pha, đất thịt thoát nước tốt",
        "climate_requirement": "Khí hậu khô mát (15 - 20°C), lượng mưa thấp (40 - 80 mm), độ ẩm thấp (< 25%)."
    },
    "kidneybeans": {
        "vi": "Đậu tây",
        "scientific": "Phaseolus vulgaris",
        "category": "Cây họ đậu",
        "soil_type": "Đất thịt pha mùn, pH 5.5 - 6.5, nhu cầu kali cao",
        "climate_requirement": "Nhiệt độ 15 - 24°C, lượng mưa 60 - 150 mm, độ ẩm trung bình."
    },
    "pigeonpeas": {
        "vi": "Đậu triều",
        "scientific": "Cajanus cajan",
        "category": "Cây họ đậu",
        "soil_type": "Đất cát pha, đất phù sa cổ, chịu hạn cao",
        "climate_requirement": "Nhiệt độ ấm (25 - 35°C), lượng mưa 90 - 180 mm, thích nghi vùng bán khô hạn."
    },
    "mothbeans": {
        "vi": "Đậu bướm",
        "scientific": "Vigna aconitifolia",
        "category": "Cây họ đậu",
        "soil_type": "Đất cát, đất đồi nghèo dinh dưỡng",
        "climate_requirement": "Chịu nhiệt độ cao (25 - 35°C), lượng mưa rất thấp (30 - 70 mm), độ ẩm thấp."
    },
    "mungbean": {
        "vi": "Đậu xanh",
        "scientific": "Vigna radiata",
        "category": "Cây họ đậu",
        "soil_type": "Đất thịt nhẹ, đất phù sa ven sông thoát nước tốt",
        "climate_requirement": "Nhiệt độ 25 - 32°C, độ ẩm cao (80 - 90%), lượng mưa 40 - 60 mm."
    },
    "blackgram": {
        "vi": "Đậu đen",
        "scientific": "Vigna mungo",
        "category": "Cây họ đậu",
        "soil_type": "Đất thịt pha sét nhẹ, giữ ẩm tốt",
        "climate_requirement": "Nhiệt độ 25 - 35°C, độ ẩm 60 - 75%, lượng mưa 60 - 80 mm."
    },
    "lentil": {
        "vi": "Đậu lăng",
        "scientific": "Lens culinaris",
        "category": "Cây họ đậu",
        "soil_type": "Đất cát pha, đất thịt nhẹ, pH 6.0 - 7.5",
        "climate_requirement": "Khí hậu ôn hòa mát mẻ (18 - 28°C), lượng mưa thấp (40 - 55 mm)."
    },
    "pomegranate": {
        "vi": "Lựu",
        "scientific": "Punica granatum",
        "category": "Cây ăn quả",
        "soil_type": "Đất thịt pha cát, đất phù sa tầng dày, thoát nước tốt",
        "climate_requirement": "Nhiệt độ 18 - 25°C, lượng mưa 100 - 115 mm, độ ẩm trung bình."
    },
    "banana": {
        "vi": "Chuối",
        "scientific": "Musa acuminata",
        "category": "Cây ăn quả",
        "soil_type": "Đất phù sa giàu mùn, nhu cầu đạm (N) và kali (K) rất cao",
        "climate_requirement": "Nhiệt độ 25 - 30°C, độ ẩm không khí 75 - 85%, lượng mưa 90 - 120 mm."
    },
    "mango": {
        "vi": "Xoài",
        "scientific": "Mangifera indica",
        "category": "Cây ăn quả lâu năm",
        "soil_type": "Đất thịt pha cát, đất đỏ bazan tầng dày",
        "climate_requirement": "Nhiệt độ 27 - 35°C, cần mùa khô phân hóa mầm hoa, lượng mưa 90 - 100 mm."
    },
    "grapes": {
        "vi": "Nho",
        "scientific": "Vitis vinifera",
        "category": "Cây ăn quả",
        "soil_type": "Đất cát pha, đất sỏi đá thoát nước tốt, nhu cầu lân (P) và kali (K) cao",
        "climate_requirement": "Khí hậu khô ráo, nắng nhiều, nhiệt độ 10 - 40°C, lượng mưa thấp (65 - 75 mm)."
    },
    "watermelon": {
        "vi": "Dưa hấu",
        "scientific": "Citrullus lanatus",
        "category": "Cây ăn quả ngắn ngày",
        "soil_type": "Đất cát pha bãi bồi, đất thịt nhẹ, pH 6.0 - 7.0",
        "climate_requirement": "Nhiệt độ 24 - 28°C, độ ẩm 80 - 90%, lượng mưa 40 - 60 mm."
    },
    "muskmelon": {
        "vi": "Dưa lưới",
        "scientific": "Cucumis melo",
        "category": "Cây ăn quả ngắn ngày",
        "soil_type": "Đất cát pha giàu hữu cơ, thoát nước tốt",
        "climate_requirement": "Khí hậu nắng ấm, nhiệt độ 26 - 30°C, lượng mưa thấp (20 - 30 mm), độ ẩm 90 - 95%."
    },
    "apple": {
        "vi": "Táo",
        "scientific": "Malus domestica",
        "category": "Cây ăn quả ôn đới",
        "soil_type": "Đất thịt pha sét giàu mùn, nhu cầu P và K đặc biệt cao",
        "climate_requirement": "Khí hậu ôn đới mát lạnh (20 - 24°C), độ ẩm 90 - 95%, lượng mưa 100 - 125 mm."
    },
    "orange": {
        "vi": "Cam",
        "scientific": "Citrus sinensis",
        "category": "Cây ăn quả có múi",
        "soil_type": "Đất thịt nhẹ, đất phù sa cổ tơi xốp, kỵ úng nước",
        "climate_requirement": "Nhiệt độ 15 - 35°C, độ ẩm 90 - 95%, lượng mưa 100 - 120 mm."
    },
    "papaya": {
        "vi": "Đu đủ",
        "scientific": "Carica papaya",
        "category": "Cây ăn quả",
        "soil_type": "Đất giàu mùn, thoát nước nhanh, không ngập úng",
        "climate_requirement": "Nhiệt độ 25 - 35°C, độ ẩm 90 - 95%, lượng mưa 140 - 250 mm."
    },
    "coconut": {
        "vi": "Dừa",
        "scientific": "Cocos nucifera",
        "category": "Cây công nghiệp nhiệt đới",
        "soil_type": "Đất cát pha, đất phù sa ven biển hoặc bờ kênh rạch",
        "climate_requirement": "Nhiệt độ 25 - 30°C, độ ẩm cao (90 - 100%), lượng mưa dồi dào (130 - 230 mm)."
    },
    "cotton": {
        "vi": "Bông",
        "scientific": "Gossypium hirsutum",
        "category": "Cây công nghiệp lấy sợi",
        "soil_type": "Đất phù sa nhẹ, đất thịt pha sét thoát nước",
        "climate_requirement": "Nhiệt độ 22 - 28°C, độ ẩm 75 - 85%, lượng mưa vừa phải (60 - 90 mm)."
    },
    "jute": {
        "vi": "Đay",
        "scientific": "Corchorus capsularis",
        "category": "Cây lấy sợi",
        "soil_type": "Đất phù sa hàng năm bồi đắp, giữ nước tốt",
        "climate_requirement": "Nhiệt độ cao (24 - 26°C), độ ẩm 75 - 90%, lượng mưa lớn (150 - 200 mm)."
    },
    "coffee": {
        "vi": "Cà phê",
        "scientific": "Coffea arabica / canephora",
        "category": "Cây công nghiệp lâu năm",
        "soil_type": "Đất đỏ bazan tầng canh tác sâu (> 70 cm), tơi xốp, thoát nước tốt",
        "climate_requirement": "Nhiệt độ 23 - 28°C, độ ẩm 55 - 70%, lượng mưa 120 - 180 mm."
    }
}

# Các tập thông số mẫu theo vùng sinh thái
SAMPLE_PRESETS = {
    "Vùng đồng bằng phù sa (Chuyên canh lúa nước)": {
        "N": 85.0, "P": 45.0, "K": 40.0,
        "temperature": 23.5, "humidity": 82.0, "ph": 6.5, "rainfall": 230.0
    },
    "Vùng đồi núi bán khô hạn (Cây họ đậu chịu hạn)": {
        "N": 15.0, "P": 20.0, "K": 25.0,
        "temperature": 36.0, "humidity": 30.0, "ph": 7.2, "rainfall": 40.0
    },
    "Vùng cao nguyên đất đỏ (Cây công nghiệp lâu năm)": {
        "N": 105.0, "P": 30.0, "K": 32.0,
        "temperature": 25.0, "humidity": 65.0, "ph": 6.8, "rainfall": 160.0
    },
    "Vùng cận ôn đới (Cây ăn quả cần lân và kali cao)": {
        "N": 25.0, "P": 130.0, "K": 200.0,
        "temperature": 21.0, "humidity": 92.0, "ph": 6.0, "rainfall": 110.0
    },
    "Vùng phù sa ven sông (Cây ăn quả nhiệt đới)": {
        "N": 100.0, "P": 75.0, "K": 50.0,
        "temperature": 27.0, "humidity": 80.0, "ph": 6.2, "rainfall": 95.0
    }
}
