import { OpenAI } from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Anthropic from '@anthropic-ai/sdk';

export type LLMProvider = 'openai' | 'google' | 'anthropic';

export interface LLMRequest {
    systemPrompt?: string;
    prompt: string;
    temperature?: number;
    maxTokens?: number;
    responseFormat?: 'json_object' | 'text';
}

export interface LLMResponse {
    content: string;
    model: string;
    usage?: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
    };
}

export async function generateStructuredOutput<T>(
    provider: LLMProvider,
    model: string,
    request: LLMRequest
): Promise<T> {
    const rawResponse = await callLLM(provider, model, {
        ...request,
        responseFormat: 'json_object'
    });

    try {
        // Some models might wrap JSON in markdown blocks
        let cleanContent = rawResponse.content.trim();
        if (cleanContent.startsWith('```json')) {
            cleanContent = cleanContent.replace(/^```json\n/, '').replace(/\n```$/, '');
        } else if (cleanContent.startsWith('```')) {
            cleanContent = cleanContent.replace(/^```\n/, '').replace(/\n```$/, '');
        }
        
        return JSON.parse(cleanContent) as T;
    } catch (error) {
        console.error(`Failed to parse ${provider} response as JSON:`, rawResponse.content);
        throw new Error(`LLM ${provider} JSON parse error: ${error instanceof Error ? error.message : String(error)}`);
    }
}

async function callLLM(
    provider: LLMProvider,
    model: string,
    request: LLMRequest
): Promise<LLMResponse> {
    switch (provider) {
        case 'openai':
            return callOpenAI(model, request);
        case 'google':
            return callGoogle(model, request);
        case 'anthropic':
            return callAnthropic(model, request);
        default:
            throw new Error(`Unsupported LLM provider: ${provider}`);
    }
}

async function callOpenAI(model: string, request: LLMRequest): Promise<LLMResponse> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error('OPENAI_API_KEY is missing');

    const openai = new OpenAI({ apiKey });
    const response = await openai.chat.completions.create({
        model,
        messages: [
            ...(request.systemPrompt ? [{ role: 'system' as const, content: request.systemPrompt }] : []),
            { role: 'user' as const, content: request.prompt }
        ],
        temperature: request.temperature ?? 0.7,
        max_tokens: request.maxTokens,
        response_format: request.responseFormat === 'json_object' ? { type: 'json_object' } : undefined
    });

    return {
        content: response.choices[0]?.message?.content ?? '',
        model: response.model,
        usage: {
            promptTokens: response.usage?.prompt_tokens ?? 0,
            completionTokens: response.usage?.completion_tokens ?? 0,
            totalTokens: response.usage?.total_tokens ?? 0
        }
    };
}

async function callGoogle(model: string, request: LLMRequest): Promise<LLMResponse> {
    const apiKey = process.env.GOOGLE_AI_API_KEY || process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GOOGLE_AI_API_KEY or GEMINI_API_KEY is missing');

    const genAI = new GoogleGenerativeAI(apiKey);
    const genModel = genAI.getGenerativeModel({ model });

    const prompt = request.systemPrompt 
        ? `${request.systemPrompt}\n\nUser: ${request.prompt}`
        : request.prompt;

    const result = await genModel.generateContent({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
            temperature: request.temperature ?? 0.7,
            maxOutputTokens: request.maxTokens,
            responseMimeType: request.responseFormat === 'json_object' ? 'application/json' : 'text/plain'
        }
    });

    const response = await result.response;
    const text = response.text();

    return {
        content: text,
        model: model,
        usage: {
            promptTokens: response.usageMetadata?.promptTokenCount ?? 0,
            completionTokens: response.usageMetadata?.candidatesTokenCount ?? 0,
            totalTokens: response.usageMetadata?.totalTokenCount ?? 0
        }
    };
}

async function callAnthropic(model: string, request: LLMRequest): Promise<LLMResponse> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error('ANTHROPIC_API_KEY is missing');

    const anthropic = new Anthropic({ apiKey });
    
    // Anthropic doesn't have a native "JSON mode" like OpenAI, 
    // but they handle system prompts well.
    const prompt = request.responseFormat === 'json_object'
        ? `${request.prompt}\n\nIMPORTANT: Return ONLY valid JSON.`
        : request.prompt;

    const response = await anthropic.messages.create({
        model,
        system: request.systemPrompt,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: request.maxTokens ?? 4096,
        temperature: request.temperature ?? 0.7
    });

    const content = response.content[0].type === 'text' ? response.content[0].text : '';

    return {
        content,
        model: response.model,
        usage: {
            promptTokens: response.usage.input_tokens,
            completionTokens: response.usage.output_tokens,
            totalTokens: response.usage.input_tokens + response.usage.output_tokens
        }
    };
}
