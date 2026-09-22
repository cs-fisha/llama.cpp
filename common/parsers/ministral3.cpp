#include "parsers.h"

common_chat_params common_chat_params_init_ministral_3(const common_chat_template &          tmpl,
                                                       const autoparser::generation_params & inputs) {
    common_chat_params data;

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto role = msg.value("role", "");
        if (role != "system" && role != "assistant") {
            // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
            adjusted_messages.push_back(msg);
            continue;
        }

        auto content = json::array();

        // If message contains `reasoning_content`, add it as a block of type `thinking`
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            content.push_back({
                { "type",     "thinking"                                     },
                { "thinking", msg.at("reasoning_content").get<std::string>() },
            });
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            if (msg.at("content").is_string()) {
                content.push_back({
                    { "type", "text"                               },
                    { "text", msg.at("content").get<std::string>() },
                });
            } else if (msg.at("content").is_array()) {
                auto blocks = msg.at("content");
                content.insert(blocks);
            }
        }

        auto adjusted       = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        adjusted_messages.push_back(adjusted);
    }

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = true;

    data.supports_thinking  = true;
    data.thinking_start_tag = "[THINK]";
    data.thinking_end_tags  = { "[/THINK]" };
    data.prompt = common_chat_template_direct_apply_impl(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.generation_prompt =
        common_chat_template_generation_prompt_impl(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    if (inputs.has_continuation()) {
        const auto & msg = inputs.continue_msg;

        data.generation_prompt = "[THINK]" + msg.reasoning_content;
        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += "[/THINK]" + msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.eps();
        auto reasoning =
            extract_reasoning ? p.optional("[THINK]" + p.reasoning(p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (has_response_format) {
            // Ministral wants to emit json surrounded by code fences
            return generation_prompt +
                   (reasoning << "```json" << p.content(p.schema(p.json(), "response-format", inputs.json_schema))
                              << "```");
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto   schema   = common_chat_tool_parameters(function);

                tool_choice |=
                    p.rule("tool-" + name, p.tool_open(p.tool_name(p.literal(name)) + "[ARGS]") +
                                               p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));
            });

            auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat("[TOOL_CALLS]" + tool_choice, min_calls, max_calls));
            // Obligatory tool calls for the "tool call during reasoning" path (min 1),
            // so a mid-thought mention of [TOOL_CALLS] that is not a real call fails and
            // falls back to the closed-[/THINK] path (see deepseek / #29248).
            auto tool_calls_required =
                p.trigger_rule("tool-call-required", p.repeat("[TOOL_CALLS]" + tool_choice, 1, max_calls));

            auto content_and_tools = reasoning << p.content(p.until("[TOOL_CALLS]")) << tool_calls;
            if (!extract_reasoning) {
                return generation_prompt + content_and_tools;
            }

            // Model may emit [TOOL_CALLS] inside [THINK]... without [/THINK] first (#29248).
            // Prefer this branch when it fully consumes the input; otherwise fall back so a
            // mid-thought [TOOL_CALLS] mention that continues after the JSON still uses the
            // closed-[/THINK] path (see existing "fake tool call marker in reasoning" test).
            auto reasoning_with_tc = "[THINK]" + p.reasoning(p.until_one_of({ "[/THINK]", "[TOOL_CALLS]" })) +
                                     p.optional(p.literal("[/THINK]")) + tool_calls_required;

            return generation_prompt + ((reasoning_with_tc + p.end()) | content_and_tools);
        }

        // Content only parser
        include_grammar = false;
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar(
            [&](const common_grammar_builder & builder) { parser.build_grammar(builder, data.grammar_lazy); });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]" }
        };
    }

    return data;
}
