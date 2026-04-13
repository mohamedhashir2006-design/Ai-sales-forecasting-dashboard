def recommend_action(row, sales_trend):

    # Base logic (use existing columns)
    if row["response_rate"] > 0.8:
        action = "📞 Call immediately"

    elif row["response_rate"] < 0.2:
        action = "💸 Offer discount"

    else:
        action = "✉️ Email follow-up"

    # Add urgency if trend is down
    if sales_trend == "down":
        action = "🔥 " + action

    return action