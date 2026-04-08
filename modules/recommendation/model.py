def recommend_action(row, sales_trend):
    if sales_trend == "down":
        return "🔥 Urgent: Call customer"

    if row["lead_score"] > 85:
        return "📞 Call immediately"

    elif row["conversion_rate"] < 0.2:
        return "💸 Offer discount"

    else:
        return "📧 Email follow-up"